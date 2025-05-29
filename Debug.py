from ImagePrep import preprocess_image, preprocess_csv

test_csv = r'C:\Users\17317\PycharmProjects\Classic LeNetCNN\data\test.csv'

tensor = preprocess_csv(test_csv)
print(f"CSV tensor shape: {tensor.shape}")  # Should be [N, 1, 28, 28]
print(f"Tensor dtype: {tensor.dtype}")  # Should be torch.float32
print(f"Value range: ({tensor.min():.3f}, {tensor.max():.3f})")  # Should be between 0 and 1