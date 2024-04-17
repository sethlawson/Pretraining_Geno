import json

# Path to the JSON file

for gpt in ['4', '500']:

    file_path = f'embeddings_before_tsne{gpt}.json'

    # Read the JSON file and load its contents into a dictionary
    with open(file_path, 'r') as json_file:
        data_dict = json.load(json_file)

    # Now, data_dict contains the contents of the JSON file as a dictionary
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Step 1: Combine all embeddings into a single array and keep track of keys
    embeddings = []
    keys = []

    for key, embedding_list in data_dict.items():
        embeddings.extend(embedding_list)
        keys.extend([key] * len(embedding_list))

    # Convert lists to numpy arrays
    embeddings = np.array(embeddings)
    keys = np.array(keys)

    # Step 2: Apply PCA to reduce the dimensionality to 50 components
    pca = PCA(n_components=50)
    embeddings_pca = pca.fit_transform(embeddings)

    # Step 3: Perform t-SNE dimensionality reduction on the PCA-transformed embeddings
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_pca_tsne = tsne.fit_transform(embeddings_pca)

    # Step 4: Plot the t-SNE results, coloring points based on associated key
    plt.figure(figsize=(10, 8))
    unique_keys = np.unique(keys)

    # Define keys to plot
    keys_to_plot = ['lncRNA', 'rRNA', 'telomerase_RNA', 'SRP_RNA', 'RNase_P_RNA', 'RNase_MRP_RNA', 'circRNA', 'miRNA', 
                    'pre_miRNA', 'precursor_RNA', 'piRNA', 'rasiRNA', 'scRNA', 'snRNA', 'tmRNA', 'scaRNA', 
                    'snoRNA', 'tRNA', 'siRNA', 'vault_RNA', 'Y_RNA', 'hammerhead_ribozyme']

    # Create Scatter plots for each key
    plt.figure(figsize=(10, 8))
    for key in keys_to_plot:
        mask = keys == key
        plt.scatter(embeddings_pca_tsne[mask, 0], embeddings_pca_tsne[mask, 1], label=key, s=2)

    # Create layout
    plt.title('t-SNE Plot of PCA-transformed Embeddings (50 components)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Place the legend outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the result as an HTML file
    plt.savefig(f'Figures/tsne_embeddings_{gpt}.svg',bbox_inches='tight')
    plt.savefig(f'Figures/tsne_embeddings_{gpt}.png', dpi=500, bbox_inches='tight')
    plt.close()

    # Define long and short RNA species
    long_rnas = ['lncRNA', 'rRNA', 'telomerase_RNA', 'SRP_RNA', 'RNase_P_RNA', 'RNase_MRP_RNA']
    short_rnas = ['circRNA', 'miRNA', 'pre_miRNA', 'precursor_RNA', 'piRNA', 'rasiRNA', 'scRNA', 'snRNA', 'tmRNA', 'scaRNA', 'snoRNA', 'tRNA', 'siRNA', 'vault_RNA', 'Y_RNA', 'hammerhead_ribozyme']

    # Create Scatter plot for all RNAs
    plt.figure(figsize=(10, 8))
    for key in keys_to_plot:
        mask = keys == key
        if key in long_rnas:
            plt.scatter(embeddings_pca_tsne[mask, 0], embeddings_pca_tsne[mask, 1], label=key, color='blue', s=2)
        elif key in short_rnas:
            plt.scatter(embeddings_pca_tsne[mask, 0], embeddings_pca_tsne[mask, 1], label=key, color='red', s=2)

    # Create layout
    plt.title('t-SNE Plot of PCA-transformed Embeddings (50 components)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    # Place the legend outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the result as an HTML file
    plt.savefig(f'Figures/tsne_embeddings_long_short_{gpt}.png', dpi=500, bbox_inches='tight')
    plt.savefig(f'Figures/tsne_embeddings_long_short_{gpt}.svg', bbox_inches='tight')
    plt.close()