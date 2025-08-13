from nets import MODELS



if __name__ == '__main__': 

    model_options = MODELS.keys()
    
    import argparse
    parser = argparse.ArgumentParser(description='Run centralized simulation')
    parser.add_argument('dataset_folder', type=str, help='Path to the dataset folder')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--net', type=str, choices=model_options, required=True, help='Model to use for simulation')
    

    args = parser.parse_args()