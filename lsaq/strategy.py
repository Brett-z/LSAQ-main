
def quant_strategy(gpu_memory, model_type):
    # if model_type == "qwen3":
    #     model_16 = 16325
    #     model_12 = 13022
    #     model_8 = 9668
    #     model_6 = 8108
    #     model_4 = 6420
    #     num_layer = 36

    # elif model_type == "llama":
    #     model_16 = 13413
    #     model_12 = 10474
    #     model_8 = 7434
    #     model_6 = 6080
    #     model_4 = 4614
    #     num_layer = 32

    if model_type == "qwen3":
        model_16 = 19000
        model_12 = 16000
        model_8 = 12000
        model_6 = 10500
        model_4 = 9000
        num_layer = 36

    elif model_type == "llama":
        model_16 = 15000
        model_12 = 12000
        model_8 = 8800
        model_6 = 7500
        model_4 = 6000
        num_layer = 32
    
    if gpu_memory > model_16:
        num_to_quant, low_bit = 0, 0
    elif gpu_memory > model_12:
        num_to_quant, low_bit = num_layer/2, 8
    elif gpu_memory > model_8:
        num_to_quant, low_bit = num_layer, 8
    elif gpu_memory > model_6:
        num_to_quant, low_bit = num_layer/2, 4
    elif gpu_memory > model_4:
        num_to_quant, low_bit = num_layer, 4
    else:
        num_to_quant, low_bit = -1, -1
     
    return int(num_to_quant), low_bit