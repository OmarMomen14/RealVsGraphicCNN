import ImageClassifier as ic
import os
import shutil


'''print('Testing a real image: is_real? >> ', ic.is_real('realImage.jpg'))
print('Testing a graphic image: is_real? >> ', ic.is_real('graphicImage.jpg'))

files_list = []
for i in os.listdir('./groupTesting/'):
    files_list.append('./groupTesting/'+i)

results_list = ic.are_real(files_list)

for (file_path, r) in zip(files_list, results_list):
    if r == True:
        shutil.move(file_path, './Results')
'''

imgURLS = [
        "http://images.assetsdelivery.com/compings_v2/choreograph/choreograph1902/choreograph190200015.jpg",
        "http://images.assetsdelivery.com/compings_v2/voyata/voyata1904/voyata190400686.jpg",
        "http://images.assetsdelivery.com/compings_v2/voyata/voyata1904/voyata190400723.jpg",
        "http://images.assetsdelivery.com/compings_v2/hiddencatch/hiddencatch1811/hiddencatch181100212.jpg",
        "http://images.assetsdelivery.com/compings_v2/choreograph/choreograph1905/choreograph190500084.jpg",
        "http://images.assetsdelivery.com/compings_v2/ryanking999/ryanking9991904/ryanking999190400458.jpg",
        "http://images.assetsdelivery.com/compings_v2/deagreez/deagreez1904/deagreez190401047.jpg",
        "http://images.assetsdelivery.com/compings_v2/choreograph/choreograph1905/choreograph190500090.jpg",
        "http://images.assetsdelivery.com/compings_v2/deagreez/deagreez1902/deagreez190202206.jpg",
        "http://images.assetsdelivery.com/compings_v2/fizkes/fizkes1810/fizkes181000669.jpg",
        "http://images.assetsdelivery.com/compings_v2/voyata/voyata1904/voyata190400718.jpg",
        "http://images.assetsdelivery.com/compings_v2/rh2010/rh20101712/rh2010171201039.jpg"
    ]

print(ic.are_real_URLS(imgURLS))