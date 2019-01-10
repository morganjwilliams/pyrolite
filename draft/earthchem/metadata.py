from bs4 import BeautifulSoup

oceanfeatures = BeautifulSoup(open("ofn.html").read()).get_text().splitlines()

with open("ocean_features.txt", "w") as fp:
    fp.write("\n".join([i.strip() for i in oceanfeatures if i.strip()]))

oceanfeatures = BeautifulSoup(open("ridges.html").read()).get_text().splitlines()

with open("ridges.txt", "w") as fp:
    fp.write("\n".join([i.strip() for i in oceanfeatures if i.strip()]))
