import shutil
import os
import pandas as pd
import audio2numpy as a2n
from scipy.io import wavfile
from pydub import AudioSegment, effects
import numpy as np

def createSectionsDict(loc):
    sections_main = {}
    sections = {}
    pallavi_start = {}
    pallavi_start_main = {}
    violin_alap_start = {}
    violin_alap_start_main = {}
    count = 0
    sec_present_count = 0
    for sec_loc in os.scandir(loc):
        if sec_loc.is_file():
            continue
        for filename in os.scandir(sec_loc.path):
            if filename.is_file():
                continue
            for filename2 in os.scandir(filename.path):
                if(filename2.path.endswith('sections-manual-p.txt')):
                    sec_present_count = 1
                    section_data = pd.read_csv(filename2, sep = '\t',names=['Start','Number','Time','Section'])
                    sections.update({filename.name:section_data})

                    for i in range(len(section_data['Section'])):
                        if(section_data['Section'][i]=='Violin ālāp'):
                            violin_alap_start.update({filename.name:section_data['Start'][i]})
                            count+=1
                        elif(section_data['Section'][i]=='Pallavi'):
                            pallavi_start.update({filename.name:section_data['Start'][i]})
                            count+=1
                            break
                    
                    if(count != 2):
                        pallavi_start.update({filename.name:0})
                        violin_alap_start.update({filename.name:0})
                    count = 0
                
            if(sec_present_count == 0):
                pallavi_start.update({filename.name:0})
                violin_alap_start.update({filename.name:0})
            sec_present_count = 0
        sections_main.update({sec_loc.name:sections})
        sections = {}
        pallavi_start_main.update({sec_loc.name:pallavi_start})
        pallavi_start = {}
        violin_alap_start_main.update({sec_loc.name:violin_alap_start})
        violin_alap_start = {}
    return sections_main, pallavi_start_main, violin_alap_start_main

def writeSolos(
        new_loc, 
        data_folder, 
        pallavi_start_main, 
        violin_alap_start_main
    ):
    data_num = 0
    for concerts in os.scandir(new_loc):
        if concerts.is_file():
            continue
        for songs in os.scandir(concerts.path):
            if songs.is_file():
                continue
            for multitracks in os.scandir(songs.path):     
                if(multitracks.name.endswith('.multitrack-violin.mp3')):
                    audio_data = AudioSegment.from_file(multitracks.path, "mp3")
                    cut_point1 = violin_alap_start_main[concerts.name][songs.name] * 1000 # ms
                    cut_point2 = pallavi_start_main[concerts.name][songs.name] * 1000 # ms
                    if cut_point1 != 0 and cut_point2 != 0:
                        violin_solo = audio_data[cut_point1:cut_point2]
                        print(f'Concert: {concerts.name}')
                        print(f'Song: {songs.name}')
                        violin_solo = effects.normalize(violin_solo)
                        violin_solo.export(new_loc + data_folder + '/' + 'violin_solo' + str(data_num) + '.wav', format = 'wav')
                        data_num+=1

if __name__ == "__main__":
    loc = '../saraga/dataset/carnatic/'
    _, pallavi_start_main, violin_alap_start_main = createSectionsDict(loc)
    new_loc = '../Datasets/carnatic/' # Write the folder where the carnatic dataset exists
    data_folder = 'violin_solo_dataset'
    new_path = os.path.join(new_loc, data_folder)
    if(not(os.path.isdir(new_path))):
        os.mkdir(new_path)
    writeSolos(new_loc, data_folder, pallavi_start_main, violin_alap_start_main)

    # Janakipathe: Violin solo start: 825 ms
    # Manasaramathi: Violin solo start: 572 ms

    # Rewriting the last two
    last_loc = '../Datasets/carnatic/KP Nandini at Arkay by KP Nandini/'
    loc1 = 'Janakipathe Jaya Karunya Jaladhe/Janakipathe Jaya Karunya Jaladhe.multitrack-violin.mp3'
    loc2 = 'Manasaramathi/Manasaramathi.multitrack-violin.mp3'
    
    audio_data = AudioSegment.from_file(last_loc+loc1, "mp3")
    slugs = 'multitrack-violin'
    fs = 44100
    cut_point1 = 820 * 1000 # ms
    cut_point2 = (820+385.151) * 1000 # ms
    violin_solo = audio_data[cut_point1:cut_point2]
    violin_solo = effects.normalize(violin_solo)
    violin_solo.export(new_loc + data_folder + '/' + 'violin_solo' + str(11) + '.wav', format = 'wav')

    audio_data = AudioSegment.from_file(last_loc+loc2, "mp3")
    slugs = 'multitrack-violin'
    fs = 44100
    cut_point1 = 570 * 1000 # ms
    cut_point2 = (570+305) * 1000 # ms
    violin_solo = audio_data[cut_point1:cut_point2]
    violin_solo = effects.normalize(violin_solo)
    violin_solo.export(new_loc + data_folder + '/' + 'violin_solo' + str(12) + '.wav', format = 'wav')
        
