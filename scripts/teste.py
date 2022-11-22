import keyboard
import os

os.system('sudo')

while True:
    try:  # used try so that if user pressed other than the given key error will not be shown
                if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                    print('teste')
                    #reset(Empty())
                    #main()
    except:
        continue