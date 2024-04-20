import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

EXPERTS_PER_LAYER = 8

BASE_DIR = "logs/prompt1/"


class Expert:
    def __init__(self, uid0, uid1, eviction_group, offloaded, index):
        self.uid0 = uid0
        self.uid1 = uid1
        self.eviction_group = eviction_group
        self.offloaded = offloaded
        self.index = index


class Layer:
    def __init__(self):
        self.activated_experts = []
        self.offloaded_experts = []


class TokenTrace:
    def __init__(self, token):
        self.token = token
        self.layers = [Layer() for _ in range(32)]


def create_layers_and_token_traces(experts, tokens):
    token_traces = []
    current_layer_index = 0
    token_index = 0

    for i in range(0, len(experts), 6):  # 6 experts per layer
        layer = Layer()

        # Create activated experts
        for j in range(i, i + 2):
            if j < len(experts):
                layer.activated_experts.append(experts[j])

        # Create offloaded experts
        for j in range(i + 2, i + 6):
            if j < len(experts):
                layer.offloaded_experts.append(experts[j])

        # Add the layer to the current TokenTrace or create a new one
        if current_layer_index % 32 == 0:
            token_trace = TokenTrace(tokens[token_index])
            token_traces.append(token_trace)
            token_index += 1

        token_trace.layers[current_layer_index % 32] = layer
        current_layer_index += 1

    return token_traces


def read_tsv(file_path):
    experts = []

    with open(file_path, "r") as file:
        lines = file.readlines()

        # Remove unwanted lines
        lines = [line for line in lines if not line.startswith("load_experts called:")]

        # Parse the lines and populate the Expert objects
        for line in lines[1:]:  # Skip the header line
            uid0, uid1, eviction_group, offloaded, index = line.strip().split("\t")
            expert = Expert(
                uid0, uid1, int(eviction_group), offloaded == "True", int(index)
            )
            experts.append(expert)

    return experts


def read_tokens(file_path):
    with open(file_path, "r") as file:
        tokens = [line.strip() for line in file.readlines()]
    return tokens


def parse_logs():
    # Example usage

    file_path = BASE_DIR + "expert_cache.tsv"
    token_file_path = BASE_DIR + "generated_tokens.txt"

    experts = read_tsv(file_path)
    tokens = read_tokens(token_file_path)
    token_traces = create_layers_and_token_traces(experts, tokens)

    # Print the generated TokenTraces
    # for i, token_trace in enumerate(token_traces):
    #     print(f"TokenTrace {i + 1}: {token_trace.token}")
    #     for j, layer in enumerate(token_trace.layers):
    #         print(f"  Layer {j + 1}:")
    #         print(
    #             f"    Activated Experts: {[expert.uid1 for expert in layer.activated_experts]}"
    #         )
    #         print(
    #             f"    Offloaded Experts: {[expert.uid1 for expert in layer.offloaded_experts]}"
    #         )
    return tokens, token_traces


def plot_layer(layer_num, tokens, active_experts, cached_experts):
    print(f"Plotting layer {layer_num}...")
    tensor = np.zeros((len(tokens), EXPERTS_PER_LAYER), dtype=int)
    for token_idx in range(len(tensor)):
        for eid in active_experts[token_idx]:
            tensor[token_idx][eid] += 1  # Red for active experts
        for eid in cached_experts[token_idx]:
            tensor[token_idx][eid] += 2  # Blue for cached experts

    # Transpose the tensor to match the desired plot orientation
    tensor = tensor.T

    # Define the color map with the correct colors
    cmap = matplotlib.colors.ListedColormap(["white", "red", "lightblue", "purple"])
    # plt.figure(figsize=(len(tokens) * 0.1, EXPERTS_PER_LAYER * 0.3))
    plt.imshow(tensor, cmap=cmap, interpolation="nearest", aspect="auto")

    # Create custom color patches and labels
    patch1 = mpatches.Patch(color="red", label="Activated (Miss)")
    patch2 = mpatches.Patch(color="lightblue", label="Cached")
    patch3 = mpatches.Patch(color="purple", label="Activated (Hit)")

    # Create legend from custom color patches
    plt.legend(
        handles=[patch1, patch2, patch3],
        loc="upper right",
        bbox_to_anchor=(1, 1),
    )

    plt.title(f"Layer {layer_num} Activation and Caching")
    plt.xlabel("Generated Tokens")
    plt.ylabel("Expert ID (0-7)")

    # Set x-axis labels to tokens and y-axis labels to expert IDs
    plt.xticks(np.arange(len(tokens)), tokens, rotation=45)
    plt.yticks(np.arange(EXPERTS_PER_LAYER), np.arange(EXPERTS_PER_LAYER))

    plt.tight_layout()
    # Save the plot
    plt.savefig(BASE_DIR + f"layer{layer_num}.png")
    # Show the plot
    # plt.show()
    plt.close()


tokens, token_traces = parse_logs()
print(f"number of tokens: {len(tokens)}")
print(f"number of token_traces: {len(token_traces)}")

# Plot and save the first layer of all TokenTraces


for layer_num in range(32):
    active_experts = []
    cached_experts = []
    num_token_to_plot = len(tokens)
    token_traces = token_traces[:num_token_to_plot]
    tokens = tokens[:num_token_to_plot]
    for token_trace in token_traces:
        layer = token_trace.layers[layer_num]
        curr_active_experts = []
        curr_cached_experts = [i for i in range(EXPERTS_PER_LAYER)]
        for expert in layer.activated_experts:
            curr_active_experts.append(int(expert.uid1))
        for expert in layer.offloaded_experts:
            curr_cached_experts.remove(int(expert.uid1))
        active_experts.append(curr_active_experts)
        cached_experts.append(curr_cached_experts)
    plot_layer(layer_num, tokens, active_experts, cached_experts)
