import os

def get_unique_filename(base_name, extension=".csv"):
    """
    Generate a unique file name by appending an incremental number if the file exists.
    
    Args:
        base_name (str): The base name of the file without extension.
        extension (str): The file extension (default is ".csv").
    
    Returns:
        str: A unique file name with the format "base_name.csv", "base_name_1.csv", etc.
    """
    filename = f"{base_name}{extension}"
    counter = 1
    
    while os.path.exists(filename):
        filename = f"{base_name}_{counter}{extension}"
        counter += 1
    
    return filename
