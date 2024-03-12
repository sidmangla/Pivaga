import os
dir_ =  "/home/sid/Documents/new_pro/v1/"

for i in os.listdir(dir_):
	if i[-4:] == ".jpg":
		txt_file = str(i.replace(".jpg",".txt"))
		if not os.path.exists(dir_+txt_file):
			print(i)
			with open(os.path.join(dir_,txt_file),'w') as f:
				pass

