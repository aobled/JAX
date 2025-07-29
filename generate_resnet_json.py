import json

def generate_resnet_json(num_blocks_per_group):
    assert num_blocks_per_group in [3, 5, 7, 9], "Standard ResNet has 3, 5, 7, or 9 blocks per group"
    layers = []

    def conv_bn_relu(id_prefix, input_id, filters, strides=(1, 1)):
        conv_id = f"{id_prefix}_conv"
        bn_id = f"{id_prefix}_bn"
        relu_id = f"{id_prefix}_relu"
        return [
            { "type": "conv", "id": conv_id, "inputs": [input_id], "filters": filters, "kernel_size": [3, 3], "strides": list(strides), "padding": "SAME" },
            { "type": "batch_norm", "id": bn_id, "inputs": [conv_id] },
            { "type": "relu", "id": relu_id, "inputs": [bn_id] }
        ], relu_id

    def residual_block(group, block, input_id, filters, downsample=False):
        prefix = f"g{group}_b{block}"
        layers_block = []

        # Main path
        strides = (2, 2) if downsample else (1, 1)
        conv1, id1 = conv_bn_relu(f"{prefix}_c1", input_id, filters, strides)
        conv2_id = f"{prefix}_c2_conv"
        bn2_id = f"{prefix}_c2_bn"
        layers_block += conv1
        layers_block.append({ "type": "conv", "id": conv2_id, "inputs": [id1], "filters": filters, "kernel_size": [3, 3], "padding": "SAME" })
        layers_block.append({ "type": "batch_norm", "id": bn2_id, "inputs": [conv2_id] })

        # Skip connection
        skip_input = input_id
        if downsample:
            proj_id = f"{prefix}_proj"
            layers_block.append({
                "type": "conv",
                "id": proj_id,
                "inputs": [input_id],
                "filters": filters,
                "kernel_size": [1, 1],
                "strides": [2, 2],
                "padding": "SAME"
            })
            skip_input = proj_id

        # Add + ReLU
        add_id = f"{prefix}_add"
        relu_out = f"{prefix}_out"
        layers_block.append({ "type": "add", "id": add_id, "inputs": [bn2_id, skip_input] })
        layers_block.append({ "type": "relu", "id": relu_out, "inputs": [add_id] })

        return layers_block, relu_out

    # Initial layers
    layers += [
        { "type": "input", "id": "input" },
        { "type": "conv", "id": "conv_init", "inputs": ["input"], "filters": 16, "kernel_size": [3, 3], "padding": "SAME" },
        { "type": "batch_norm", "id": "bn_init", "inputs": ["conv_init"] },
        { "type": "relu", "id": "relu_init", "inputs": ["bn_init"] }
    ]
    last_id = "relu_init"

    # Group 1: 16 filters
    for b in range(num_blocks_per_group):
        block, last_id = residual_block(1, b+1, last_id, 16, downsample=False)
        layers += block

    # Group 2: 32 filters
    for b in range(num_blocks_per_group):
        block, last_id = residual_block(2, b+1, last_id, 32, downsample=(b == 0))
        layers += block

    # Group 3: 64 filters
    for b in range(num_blocks_per_group):
        block, last_id = residual_block(3, b+1, last_id, 64, downsample=(b == 0))
        layers += block

    # Global average pooling and dense output
    layers += [
        { "type": "global_avg_pool", "id": "gap", "inputs": [last_id] },
        { "type": "flatten", "id": "flat", "inputs": ["gap"] },
        { "type": "dense", "id": "output", "inputs": ["flat"], "units": 10 }
    ]

    return {
        "layers": layers,
        "output_layer": "output"
    }

# Génération du ResNet32 (5 blocs par groupe)
resnet32_json = generate_resnet_json(num_blocks_per_group=5)

# Sauvegarde
output_path = "./ResNet32_CIFAR10.json"
with open(output_path, "w") as f:
    json.dump(resnet32_json, f, indent=2)

output_path
