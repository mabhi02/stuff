import os

def songInfo():
# Get the current script directory
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Set the path to the folder
    folder_path = os.path.join(script_dir, 'static/songs')

    # Check if the directory exists
    if os.path.exists(folder_path):
        # Get the list of files in the folder
        files_in_folder = os.listdir(folder_path)

        # Filter out only MP3 files
        mp3_files = [file for file in files_in_folder if file.lower().endswith('.mp3')]

        # Check if there is exactly one MP3 file
        if len(mp3_files) == 1:
            ename = mp3_files[0]
            print(f'The single MP3 file in the folder is: {ename}')
        else:
            print('Either there is no MP3 file or there are multiple MP3 files in the folder.')
            ename = None
    else:
        print(f'The directory {folder_path} does not exist.')
        ename = None
    return ename

print(songInfo())