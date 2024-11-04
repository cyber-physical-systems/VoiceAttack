from nilmtk.dataset_converters import convert_redd
import os

for house_num in range(4,5):

    input_path = ''
    import os

    for i in range(1,23):
        file_path = input_path + 'channel_' + str(i) + '.dat'
        if os.path.exists(file_path):
            # removing the file using the os.remove() method
            os.remove(file_path)
        else:
            # file not found message
            print("File not found in the directory")

    from shutil import copyfile
    src = '' + str(house_num) + '/'
    for i in range(1,23):
        copyfile(src +'channel_' + str(i) + '.dat' , input_path+'channel_' + str(i) + '.dat')


    path = ''
  

    convert_redd('', path + '/redd.h5')