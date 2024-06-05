import os

background_list = os.listdir('Data/Background')
event_list = os.listdir('Data/Event')

for background_folder in background_list:
    file_list = os.listdir(os.path.join('Data/Background', background_folder))
    folder_path = os.path.join('Data/Background', background_folder)
    for i in range(len(file_list)):
        os.rename(os.path.join(folder_path, file_list[i]), os.path.join(folder_path, f'{background_folder}_{i+1:04}.wav'))

for event_folder in event_list:
    file_list = os.listdir(os.path.join('Data/Event', event_folder))
    folder_path = os.path.join('Data/Event', event_folder)
    for i in range(len(file_list)):
        os.rename(os.path.join(folder_path, file_list[i]), os.path.join(folder_path, f'{event_folder}_{i+1:04}.wav'))