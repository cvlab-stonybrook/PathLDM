from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import io

label_dict = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (1, 1): 3,
}

human_label_dict = {
    (0, 0): "no tumor, no til",
    (0, 1): "no tumor, til",
    (1, 0): "tumor, no til",
    (1, 1): "tumor, til",
}


class TCGADataset(Dataset):
    """Dataset with tumor presence labels in text"""

    def __init__(self, config=None):
        split = config.get("split")
        data_dir = Path(config.get("root"))
        self.crop_size = config.get("crop_size", None)
        self.p_uncond = config.get("p_uncond", 0)

        # Load .h5 dataset
        self.data_file = h5py.File(data_dir / "TCGA_BRCA_10x_448_tumor.hdf5", "r")

        # Load metadata
        arr1 = np.load(
            data_dir / f"train_test_brca_tumor/{split}.npz", allow_pickle=True
        )
        self.indices = arr1["indices"]
        self.prob_tumor = arr1["prob_tumor"].tolist()
        self.prob_til = arr1["prob_til"].tolist()

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def get_random_crop(img, size):
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        img = img[y : y + size, x : x + size]
        return img

    def __getitem__(self, idx):
        x_idx = self.indices[idx]

        tile = self.data_file["X"][x_idx]
        tile = np.array(Image.open(io.BytesIO(tile)))

        image = (tile / 127.5 - 1.0).astype(np.float32)
        if self.crop_size:
            image = self.get_random_crop(image, self.crop_size)

        # Random horizontal and vertical flips
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=0).copy()
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=1).copy()

        wsi_name = self.data_file["wsi"][x_idx].decode()
        folder_name = self.data_file["folder_name"][x_idx].decode()

        # Convert tumor infiltrating lymphocytes to levels low / mid / high and add to text prompt
        p_til = self.prob_til.get(wsi_name, {}).get(folder_name)
        p_tumor = self.prob_tumor.get(wsi_name, {}).get(folder_name)

        # Replace label with unconditional label with probability p_uncond
        if np.random.rand() < self.p_uncond or (p_til is None or p_tumor is None):
            label = 4
            human_label = "unconditional"

        if p_til is not None and p_tumor is not None:
            # Convert to binary label
            p_til = int(p_til > 0.5)
            p_tumor = int(p_tumor > 0.5)

            # Convert to label
            label = label_dict[(p_tumor, p_til)]
            human_label = human_label_dict[(p_tumor, p_til)]


        return {
            "image": image,
            "class_label": label,
            "human_label": human_label,
        }
