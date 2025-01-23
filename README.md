to use a different dataset please change the dataset_type to hourly_wages for exemple :


if __name__ == "__main__":
    # Configure dataset
    dataset_type = "hourly_wages"  # Change this to "abalone" or "hourly_wages" as needed
    dataset_paths = {
        "adult": "datasets/adult.csv",
        
        "hourly_wages": "datasets/hourly_wages.csv"
    }
    
