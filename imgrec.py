# Image Recognition Example with Tiny-ImageNet
# Performs image recognition on a subset of Tiny-ImageNet
# using the CLIP model and visualizes the results using t-SNE

import torch
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset, concatenate_datasets
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

# Initialize model and processor
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the dataset
full_dataset = load_dataset('Maysee/tiny-imagenet', split='train')

# Define specific index ranges to extract
num_subsets = 10
index_ranges = [slice(i, i+50) for i in range(0, full_dataset['train'].num_rows, full_dataset['train'].num_rows//num_subsets)]

# Load the specified ranges as Dataset objects
subsets = [full_dataset.select(range(index.start, index.stop)) for index in index_ranges]

# Concatenate the Dataset objects
dataset = concatenate_datasets(subsets)

# Function to process images and extract features
def process_images(batch):
    images = [img.convert('RGB') for img in batch['image']]
    inputs = processor(images=images, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != 'input_ids'}
    
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        embeddings = outputs
        embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)
    return {'embeddings': embeddings.cpu().numpy()}

# Apply processing to all images
dataset = dataset.map(process_images, batched=True, batch_size=16)

# Collect all embeddings
embeddings = np.vstack(dataset['embeddings'])

# Perform clustering
clustering = HDBSCAN(min_cluster_size=5, metric='euclidean').fit(embeddings)

# Dimensionality reduction with t-SNE
tsne_results = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)

# Create a custom colormap
unique_labels = np.unique(clustering.labels_)
num_colors = len(unique_labels)

# Get a colormap from matplotlib
colormap = matplotlib.cm.get_cmap('nipy_spectral')  # No need to specify the number of colors
colors = [colormap(i / (num_colors - 1)) for i in range(num_colors)]

# Visualization of t-SNE with image thumbnails
fig, ax = plt.subplots(figsize=(48, 40))
for i, img in enumerate(dataset['image']):
    img = img.resize((100, 100), Image.LANCZOS)  # Resize for visualization
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img_array = np.array(img)
    
    im = Image.fromarray(img_array)
    ax.imshow(im, extent=(tsne_results[i, 0], tsne_results[i, 0] + 1, tsne_results[i, 1], tsne_results[i, 1] + 1))

# Customize plot
ax.set_xlim(tsne_results[:, 0].min(), tsne_results[:, 0].max())
ax.set_ylim(tsne_results[:, 1].min(), tsne_results[:, 1].max())
ax.axis('off')  # Remove axis
fig.savefig('tsne_visualization_high_res.png', dpi=300, bbox_inches='tight')