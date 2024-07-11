import os


class Write():
    def write_txt(self, dictionary, path = os.getcwd(), file_name = 'results.txt'):

        textfile = open(path + '\\' + file_name, "a")
        for key, value in dictionary.items():
            textfile.write(key + ' = ' + str(value) + '\n')
        textfile.close()  