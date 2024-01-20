import os
import pandas as pd
from pydub import AudioSegment, effects
import argparse

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
        actualDirectoryForCarnatic,
        data_folder, 
        pallavi_start_main, 
        violin_alap_start_main
    ):
    data_num = 0
    for concerts in os.scandir(actualDirectoryForCarnatic):
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
                        violin_solo.export(data_folder + '/' + 'violin_solo' + str(data_num) + '.wav', format = 'wav')
                        data_num+=1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Violin Type-1 Data Creation',
                    description='Creates a .pt file storing all violin audios as torch Tensors in a list as a .pt file')

    parser.add_argument('--saraga_directory_location')
    parser.add_argument('--actual_directory_location')
    args = parser.parse_args()
    saragaDirectory = args.saraga_directory_location
    actualDirectoryForCarnatic = args.actual_directory_location

    _, pallavi_start_main, violin_alap_start_main = createSectionsDict(saragaDirectory)
    data_folder = './data/violin_solo_dataset'
    if(not(os.path.isdir(data_folder))):
        os.mkdir(data_folder)
    writeSolos(actualDirectoryForCarnatic, data_folder, pallavi_start_main, violin_alap_start_main)

    # Janakipathe: Violin solo start: 825 ms
    # Manasaramathi: Violin solo start: 572 ms

    # Rewriting the last two
    last_loc = actualDirectoryForCarnatic+'KP Nandini at Arkay by KP Nandini/'
    loc1 = 'Janakipathe Jaya Karunya Jaladhe/Janakipathe Jaya Karunya Jaladhe.multitrack-violin.mp3'
    loc2 = 'Manasaramathi/Manasaramathi.multitrack-violin.mp3'
    
    audio_data = AudioSegment.from_file(last_loc+loc1, "mp3")
    slugs = 'multitrack-violin'
    fs = 44100
    cut_point1 = 820 * 1000 # ms
    cut_point2 = (820+385.151) * 1000 # ms
    violin_solo = audio_data[cut_point1:cut_point2]
    violin_solo = effects.normalize(violin_solo)
    violin_solo.export(data_folder + '/' + 'violin_solo' + str(11) + '.wav', format = 'wav')

    audio_data = AudioSegment.from_file(last_loc+loc2, "mp3")
    slugs = 'multitrack-violin'
    fs = 44100
    cut_point1 = 570 * 1000 # ms
    cut_point2 = (570+305) * 1000 # ms
    violin_solo = audio_data[cut_point1:cut_point2]
    violin_solo = effects.normalize(violin_solo)
    violin_solo.export(data_folder + '/' + 'violin_solo' + str(12) + '.wav', format = 'wav')
        
