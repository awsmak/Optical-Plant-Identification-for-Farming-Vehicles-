import os
import shutil
import random
from PIL import Image
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

class YOLOtoVOCConverter:
    def __init__(self, data_dir, output_dir, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        
        # Input directories
        self.images_dir = os.path.join(data_dir, 'raw_data', 'images')
        self.labels_dir = os.path.join(data_dir, 'raw_data', 'annotations')
        self.classes_file = os.path.join(self.labels_dir, 'classes.txt')
        
        # Output directories structure
        self.voc_dir = os.path.join(output_dir, 'VOC')
        self.image_sets_dir = os.path.join(self.voc_dir, 'ImageSets', 'Main')
        self.images_output_dir = os.path.join(self.voc_dir, 'JPEGImages')
        self.annotations_output_dir = os.path.join(self.voc_dir, 'Annotations')
        
        # Create output directories
        os.makedirs(self.image_sets_dir, exist_ok=True)
        os.makedirs(self.images_output_dir, exist_ok=True)
        os.makedirs(self.annotations_output_dir, exist_ok=True)
        
        # Load classes
        with open(self.classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def yolo_to_voc_coordinates(self, size, box):
        """Convert YOLO coordinates to VOC coordinates"""
        dw = 1./size[0]
        dh = 1./size[1]
        x, y, w, h = box
        
        # Convert from relative/normalized coordinates to absolute coordinates
        x = x/dw
        w = w/dw
        y = y/dh
        h = h/dh
        
        # Calculate VOC style coordinates
        xmin = int(x - (w/2))
        xmax = int(x + (w/2))
        ymin = int(y - (h/2))
        ymax = int(y + (h/2))
        
        # Ensure coordinates are within image bounds
        xmin = max(0, xmin)
        xmax = min(size[0], xmax)
        ymin = max(0, ymin)
        ymax = min(size[1], ymax)
        
        return xmin, ymin, xmax, ymax

    def create_xml_annotation(self, image_filename, image_size, boxes, labels):
        """Create XML annotation in PASCAL VOC format"""
        root = ET.Element('annotation')
        
        # Add basic image information
        folder = ET.SubElement(root, 'folder')
        folder.text = 'VOC'
        
        filename = ET.SubElement(root, 'filename')
        filename.text = image_filename
        
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        
        width.text = str(image_size[0])
        height.text = str(image_size[1])
        depth.text = str(3)  # Assuming RGB images
        
        # Add object annotations
        for box, label_idx in zip(boxes, labels):
            obj = ET.SubElement(root, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = self.classes[int(label_idx)]
            
            bndbox = ET.SubElement(obj, 'bndbox')
            xmin, ymin, xmax, ymax = self.yolo_to_voc_coordinates(image_size, box)
            
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)
        
        # Pretty print XML
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
        return xmlstr

    def convert_dataset(self):
        """Convert the entire dataset from YOLO to VOC format"""
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.JPG', '.jpeg', '.JPEG'))]
        
        # Create list of all files
        all_files = []
        for image_file in image_files:
            base_name = os.path.splitext(image_file)[0]
            label_file = os.path.join(self.labels_dir, base_name + '.txt')
            
            if os.path.exists(label_file):
                # Copy image
                src_image = os.path.join(self.images_dir, image_file)
                dst_image = os.path.join(self.images_output_dir, image_file)
                shutil.copy2(src_image, dst_image)
                
                # Convert annotation
                image = Image.open(src_image)
                image_size = image.size
                
                boxes = []
                labels = []
                
                # Read YOLO format annotations
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        label_idx, x_center, y_center, w, h = map(float, line.strip().split())
                        boxes.append([x_center, y_center, w, h])
                        labels.append(label_idx)
                
                # Create XML annotation
                xml_content = self.create_xml_annotation(image_file, image_size, boxes, labels)
                
                # Save XML file
                xml_filename = os.path.join(self.annotations_output_dir, base_name + '.xml')
                with open(xml_filename, 'w') as f:
                    f.write(xml_content)
                
                all_files.append(base_name)
        
        return all_files

    def split_dataset(self, all_files):
        """Split dataset into train, validation, and test sets"""
        random.shuffle(all_files)
        
        num_files = len(all_files)
        num_train = int(num_files * self.train_ratio)
        num_valid = int(num_files * self.valid_ratio)
        
        train_files = all_files[:num_train]
        valid_files = all_files[num_train:num_train + num_valid]
        test_files = all_files[num_train + num_valid:]
        
        # Write split files
        splits = {
            'train': train_files,
            'val': valid_files,
            'test': test_files
        }
        
        for split_name, split_files in splits.items():
            with open(os.path.join(self.image_sets_dir, f'{split_name}.txt'), 'w') as f:
                for filename in split_files:
                    f.write(filename + '\n')
        
        return len(train_files), len(valid_files), len(test_files)

    def convert(self):
        """Run the complete conversion process"""
        print("Starting conversion from YOLO to VOC format...")
        all_files = self.convert_dataset()
        
        print("Splitting dataset...")
        train_size, valid_size, test_size = self.split_dataset(all_files)
        
        print("\nConversion completed!")
        print(f"Dataset split sizes:")
        print(f"Training: {train_size} images")
        print(f"Validation: {valid_size} images")
        print(f"Testing: {test_size} images")
        print(f"\nOutput directory: {self.voc_dir}")

def main():
    parser = argparse.ArgumentParser(description='Convert YOLO format dataset to PASCAL VOC format')
    parser.add_argument('data_dir', help='Path to the root data directory')
    parser.add_argument('output_dir', help='Path to the output directory')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training set ratio (default: 0.7)')
    parser.add_argument('--valid-ratio', type=float, default=0.2, help='Validation set ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio (default: 0.1)')
    
    args = parser.parse_args()
    
    converter = YOLOtoVOCConverter(
        args.data_dir,
        args.output_dir,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio
    )
    converter.convert()

if __name__ == '__main__':
    main()