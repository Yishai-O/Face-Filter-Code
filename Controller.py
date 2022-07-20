import Camera as c
import Image_Recognition as ir
import time
import Filters as f
go = ""
filter = ""
keep_going = True
c.configCamera()

def choice():
    go = input("Do you want to take another picture and run facial and image recognition on it? (y/n):\n")
    if go == "y":
        keep_going = True
    elif go == "n":
        keep_going = False
        exit()
    else:
        print("That wasn't an option")
        choice()
def filter_pick():
    filter = input("Which filter do you want to use (bunny, dog, or cat)?\n")
    if filter.lower() == "bunny":
        f.use_bunny()
        # Bunny filter isn't working so great
    elif filter.lower() == "dog":
        f.use_dog()
    elif filter.lower() == "cat":
        f.use_cat()
    else:
        print("That wasn't an option")
        filter_pick()

if __name__ == "__main__":
    while keep_going:
        c.takePicture()
        ir.run_ml()
        filter_pick()
        time.sleep(2)
        choice()