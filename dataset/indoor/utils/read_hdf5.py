import blenderproc as bproc
import h5py
import numpy as np
from PIL import Image

def read_hdf5(file_path):
    # HDF5ファイルを読み込む
    with h5py.File(file_path, 'r') as f:
        # ファイル内のすべてのデータセットを表示する
        print(f"Datasets in HDF5 file '{file_path}':")
        for name in f:
            dataset = f[name]
            if isinstance(dataset, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {dataset.shape}")
                data = dataset.dtype
                print(f"Data: {data}")
                if name == "normals":
                    dataset_values = dataset[:]
                    unique_values = np.unique(dataset_values)
                    print(f"Unique values in '{name}' lenght '{len(unique_values)}' dataset:")
                    print(unique_values)
            else:
                print(f"Not a dataset: {name}")

def convert_hdf5_to_png(input_file, output_png_path):
    # Open HDF5 file in read mode
    with h5py.File(input_file, 'r') as f:
        # Read dataset values
        dataset_values = f['normals'][:]
        
        # Normalize values to range 0-255
        normalized_values = (dataset_values - np.min(dataset_values)) / (np.max(dataset_values) - np.min(dataset_values)) * 255
        normalized_values = normalized_values.astype(np.uint8)
        
        # Convert to RGB format (assuming it's a 3-channel image)
        rgb_data = np.stack([normalized_values[:,:,0], normalized_values[:,:,1], normalized_values[:,:,2]], axis=-1)
        
        # Create PIL Image object
        image = Image.fromarray(rgb_data)
        
        # Save as PNG
        image.save(output_png_path)
        
        print(f"PNG file '{output_png_path}' created successfully.")

def main():
    # Example usage
    input_hdf5_file = 'examples/basics/basic/output/0.hdf5'
    output_png_file = 'normals.png'

    #convert_hdf5_to_png(input_hdf5_file, output_png_file)
    read_hdf5(input_hdf5_file)

if __name__ == "__main__":
    main()
