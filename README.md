# Vybe
A music similarity model that finds songs that sound perceptually similar.

## Usage
This approach is most efficient when using the FMA dataset because of the given metadata for all songs, but can be used with a custom dataset as well.
If using the FMA dataset, be sure to download all the associated CSV files as they will be used to construct the library and keep track of song information.

First, run prep_data.py followed by new_index.py and new_library.py. If using a custom dataset, you will need to produce your own metadata.csv file and song list 
in order for the index and library construction to work.

## Library and Tool Choices
Python was chosen as the primary and only language because of the pre-existing libraries for audio processing and numerical computations, and because I am already very familiar with it

Python's Librosa library is what does all the heavy lifting regarding feature extraction, it is the go-to package for Music Information Retrieval (MIR) and as such is very well documented.

Python's NumPy library was used for numerical operations and the actual construction of the feature vectors because it is efficient when doing vector operations. Works seamlessly with Librosa and FAISS

FAISS was chosen for nearest-neighbor search because it is fast, scalable, and supports inner product cosine similarity calculations which are very efficient. Should the dataset grow, FAISS would ensure
that the system remains responsive.

Streamlit was used for deploying a very lightweight, but interactive demo of the product. Users can upload an audio file, select how many results they want, and watch it work in real time.

## Algorithms and Data Structure Choices
Nearest-neighbor searching was the most intuitive retrieval option given that we are returning similarity items, which is exactly what nearest neighbor is good for. It is also compatible with
vector representations which worked well with the feature vectors.

The set of features for each song are represented as fixed-length vectors which allowed for each element to be examined on its own, but also for the group as a whole to be compared against others.

## Alternatives
I briefly experimented with OpenL3 embeddings and it showed promising results. However, I chose not to use them as the primary representation because they are inherently less interpretable, have a higher computation cost and generally increased the complexity of the system. They do provide a promising direction for future work.

## Performance and Scalability
The most computationally intensive stages, dataset prep and library construction, of the system are performed offline which allows for searches to be quick and responsive. This is true even for longer songs, though they do take slightly longer to search with.

The system itself is very modular and is broken into clearly seperated components: feature extraction, library construction and indexing, and searching. This made it easy to replace individual parts if needed without having to redesign everything else.

Admittedly, cosine similarity isn't the most interpretable metric for defining similarity, but it is better than the alternatives of actual distance or learned embeddings. In most cases, the cosine similarity maps well to results.
