from . import custom_int4 as cus
from .config import scales_dict
import torch
import torch.nn as nn

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, name_in, in_features, out_features, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # w.div_(scales).round_().mul_(scales)
    w.div_(scales).round_()
    w = w.to(torch.int8)
    scales_dict[name_in] = scales
    if n_bits == 4:
        w = cus.pack_to_uint8(w, in_features, out_features)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, name_in, in_features, out_features, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    # w.div_(scales).round_().mul_(scales)
    w.div_(scales).round_()
    w = w.to(torch.int8)
    scales_dict[name_in] = scales
    if n_bits == 4:
        w = cus.pack_to_uint8(w, in_features, out_features)
    return w


class LSAQLinear(nn.Module):
    def __init__(
        self,
        name_in,
        bit_width,
        in_features,
        out_features,
        bias=True,
        quantize_output=False,
    ):
        super().__init__()
        self.name_in = name_in
        self.bit_width = bit_width
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

    def to(self, *args, **kwargs):
        super(LSAQLinear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        if self.bit_width == 4:
            weight = cus.unpack_to_int8(self.weight, self.in_features, self.out_features)
            weight = weight.to('cuda')
            weight = weight.mul(scales_dict[self.name_in])
        else:
            weight = self.weight.mul(scales_dict[self.name_in])
            
        y = torch.functional.F.linear(x, weight, self.bias)
        return y

    @staticmethod
    def from_float(
        name_in, bit, module, weight_quant="per_channel", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = LSAQLinear(
            name_in,
            bit,
            module.in_features,
            module.out_features,
            module.bias is not None,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(module.weight, name_in, module.in_features, module.out_features, bit)
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(module.weight, name_in, module.in_features, module.out_features, bit)
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"LSAQLinear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name})"


def quantize_llama_like(
    model, mlp_quant, self_attn_quant, low_bit, weight_quant="per_channel", quantize_bmm_input=False
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, LlamaMLP):      
            if low_bit == 0:
                continue
            else:
                a = ".".join(name.split(".")[:2])
                if name in mlp_quant:
                    bit = low_bit
                    print(f'{a}: {bit} bit quant')
                else:
                    if low_bit == 4:
                        bit = 8
                        print(f'{a}: {bit} bit quant')
                    elif low_bit == 8:
                        continue
            name_in = name + '.gate_proj'
            m.gate_proj = LSAQLinear.from_float(
                name_in, bit, m.gate_proj, weight_quant=weight_quant
            )
            name_in = name + '.up_proj'
            m.up_proj = LSAQLinear.from_float(
                name_in, bit, m.up_proj, weight_quant=weight_quant
            )
            name_in = name + '.down_proj'
            m.down_proj = LSAQLinear.from_float(
                name_in, bit, m.down_proj, weight_quant=weight_quant
            )
        elif isinstance(m, LlamaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            if low_bit == 0:
                    continue
            else:
                if name in self_attn_quant:
                    bit = low_bit
                else:
                    if low_bit == 4:
                        bit = 8
                    elif low_bit == 8:
                        continue
            name_in = name + '.q_proj'
            m.q_proj = LSAQLinear.from_float(
                name_in, 
                bit,  
                m.q_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            name_in = name + '.k_proj'
            m.k_proj = LSAQLinear.from_float(
                name_in, 
                bit, 
                m.k_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            name_in = name + '.v_proj'
            m.v_proj = LSAQLinear.from_float(
                name_in, 
                bit, 
                m.v_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            name_in = name + '.o_proj'
            m.o_proj = LSAQLinear.from_float(
                name_in, 
                bit, m.o_proj, weight_quant=weight_quant
            )
    return model


def quantize_qwen_like(
    model, mlp_quant, self_attn_quant, low_bit, weight_quant="per_channel", quantize_bmm_input=False
):
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention,
        Qwen3MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, Qwen3MLP):
            if low_bit == 0:
                continue
            else:
                a = ".".join(name.split(".")[:2])
                if name in mlp_quant:
                    bit = low_bit
                    print(f'{a}: {bit} bit quant')
                else:
                    if low_bit == 4:
                        bit = 8
                        print(f'{a}: {bit} bit quant')
                    elif low_bit == 8:
                        continue
            name_in = name + '.gate_proj'
            m.gate_proj = LSAQLinear.from_float(
                name_in, bit, m.gate_proj, weight_quant=weight_quant
            )
            name_in = name + '.up_proj'
            m.up_proj = LSAQLinear.from_float(
                name_in, bit, m.up_proj, weight_quant=weight_quant
            )
            name_in = name + '.down_proj'
            m.down_proj = LSAQLinear.from_float(
                name_in, bit, m.down_proj, weight_quant=weight_quant
            )
        elif isinstance(m, Qwen3Attention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            if low_bit == 0:
                    continue
            else:
                if name in self_attn_quant:
                    bit = low_bit
                else:
                    if low_bit == 4:
                        bit = 8
                    elif low_bit == 8:
                        continue
            name_in = name + '.q_proj'
            m.q_proj = LSAQLinear.from_float(
                name_in, 
                bit,  
                m.q_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            name_in = name + '.k_proj'
            m.k_proj = LSAQLinear.from_float(
                name_in, 
                bit, 
                m.k_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            name_in = name + '.v_proj'
            m.v_proj = LSAQLinear.from_float(
                name_in, 
                bit, 
                m.v_proj,
                weight_quant=weight_quant,
                quantize_output=quantize_bmm_input,
            )
            name_in = name + '.o_proj'
            m.o_proj = LSAQLinear.from_float(
                name_in, 
                bit, m.o_proj, weight_quant=weight_quant
            )
    return model

def quantize_model(model_type, model, mlp_quant, self_attn_quant, low_bit):
    if model_type == "llama":
        model = quantize_llama_like(model, mlp_quant, self_attn_quant, low_bit)
    elif model_type == "qwen3":
        model = quantize_qwen_like(model, mlp_quant, self_attn_quant, low_bit)
    else:
        sys.exit("LSAQ does not currently support the current model. Please select a different model for deployment.")
    return model