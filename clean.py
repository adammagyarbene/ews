
# import required module
# import os
# # assign directory
# train_directory = 'train/'
# test_directory = 'test/'
# valid_directory = 'valid/'


# melanoma = 'melanoma'
# nevus = 'nevus'
# seborrheic_keratosis = 'seborrheic_keratosis'


# # iterate over files in
# # that directory
# directory = train_directory + melanoma

# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         print(f)
#         os.remove(f)

import os


def delete_every_second_image(root_folder):
    # Iterate through all subfolders and files in the root_folder
    for folder_path, _, file_names in os.walk(root_folder):
        image_files = [f for f in file_names if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        if not image_files:
            continue

        # Sort the image files alphabetically to consistently delete every second image
        image_files.sort()

        for i in range(1, len(image_files), 2):
            file_to_delete = os.path.join(folder_path, image_files[i])
            try:
                os.remove(file_to_delete)
                print(f"Deleted: {file_to_delete}")
            except Exception as e:
                print(f"Error deleting {file_to_delete}: {e}")

# Example usage:
root_folder = "/Users/adammagyar/dev/workshop/files/"
delete_every_second_image(root_folder)
