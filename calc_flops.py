class ModelFLOPSCalculator:
    def __init__(self, scaling_factor=1.0) -> None:
        self.scaling_factor = scaling_factor

    def MHA(self, nh, dh, de, L):
        return 2 * nh * (dh**2) * (3 + 2*L*de)
    
    def FFNFinal(self, de, dff):
        return 2 * de * dff
    
    def FFNSub(self, de):
        return 16 * (de ** 2)
    
    def FLOPS(self, nh, dh, de, L, dff, S, B, N):
        """
        nh: number of heads
        dh: dimension of head
        de: dimension of embedding
        
        L: Average Input Sequence Length
        dff: dimension of feed forward network
        
        S: training steps
        B: batch size
        N: number of layers
        """
        return self.scaling_factor * 3 * S * B * L * ((
            self.MHA(nh, dh, de, L) + self.FFNSub(de)
        ) * N + self.FFNFinal(de, dff))
    

if __name__ == "__main__":
    flops = ModelFLOPSCalculator(4.083/1340)

    """
    Polyglot 1B config
    {
        "bos_token_id": 0,
        "eos_token_id": 0,
        "hidden_act": "gelu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 8192,
        "layer_norm_eps": 1e-05,
        "max_position_embeddings": 2048,
        "model_type": "gpt_neox",
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "rotary_emb_base": 10000,
        "rotary_pct": 0.5,
        "tie_word_embeddings": false,
        "transformers_version": "4.22.0",
        "torch_dtype": "float32",
        "use_cache": true,
        "vocab_size": 30080
    }
    """
    polyglot_1b_flops = flops.FLOPS(
        S=200000, B=1024, N=24, nh=16, dh=128, de=2048, L=256, dff=8192
    )


    """
    Polyglot 6B config
        {"hidden_act": "gelu", "architectures": ["GPTNeoXForCausalLM"], 
        "bos_token_id": 0, 
        "eos_token_id": 0, 
        "initializer_range": 0.02, 
        "layer_norm_eps": 1e-05, 
        "model_type": "gpt_neox", 
        "hidden_size": 4096, 
        "intermediate_size": 16384, 
        "num_attention_heads": 16, 
        "num_hidden_layers": 28, 
        "max_position_embeddings": 2048, 
        "rotary_pct": 0.25, 
        "rotary_emb_base": 10000, 
        "torch_dtype": "float16", 
        "use_cache": true, "vocab_size": 30080, "num_steps": "global_step320000"
    }
    """
    polyglot_6b_flops = flops.FLOPS(
        S=320000, B=256, N=28, nh=16, dh=256, de=4096, L=256, dff=16384
    )

    tflops_per_gpu = 70
    num_gpus = 256

    print("="*20 + f"Polyglot 1B with {tflops_per_gpu} TFLOPS per GPU" + "="*20)
    print(f"Total Required Flop: {polyglot_1b_flops // 10**12}T")
    print(f"Total Estimated Time (sec): {polyglot_1b_flops // (tflops_per_gpu * num_gpus * (10**12))} seconds")
    print(f"Total Time (days): {polyglot_1b_flops // (tflops_per_gpu * num_gpus * (10**12) * 24 * 60 * 60)} days")
    print(f"Actual Spent Time {98} hours = {98/24} days")

    print("="*20 + f"Polyglot 6B with {tflops_per_gpu} TFLOPS per GPU" + "="*20)
    print(f"Total Required Flop: {polyglot_6b_flops // 10**12}T")
    print(f"Total Estimated Time (sec): {polyglot_6b_flops // (tflops_per_gpu * num_gpus * (10**12))} seconds")
    print(f"Total Time (days): {polyglot_6b_flops // (tflops_per_gpu * num_gpus * (10**12) * 24 * 60 * 60)} days")
    print(f"Actual Spent Time {731} hours = {731/24} days (tflops is 35)")
