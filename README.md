A CharLSTM program suggesting new creative names on based of its scoring and datasets

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/*** about here you need to choice from
https://pytorch.org/get-started/locally/ bassed on here waither you use cpu or gpu for it
pip install numpy

well its good at creating new names but dont expect much

by the way i think i fail to use brancs correctly on github bu its need to look like this

project_root/
│
├── seeds/
│   ├── dnd_feminine.txt
│   ├── goddess_names.txt
│   ├── japanese_female.txt
│   ├── jp_name.txt
│   ├── female_names.txt (this depand whether you want this data or not )
├── finalsuggestion.txt
├── generated_long.txt
├── generated_short.txt
├── model.pt
├── nameforge_ml.py
└── test.py

i hope you get the idea
