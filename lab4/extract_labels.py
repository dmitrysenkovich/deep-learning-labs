import h5py

TRAIN_LABELS_FILE_NAME = 'train/digitStruct.mat'
TEST_LABELS_FILE_NAME = 'test/digitStruct.mat'

file = h5py.File(TRAIN_LABELS_FILE_NAME, 'r')
digitStruct = file['digitStruct']
bboxes = digitStruct['bbox']
labels = []
for i in range(len(bboxes)):
    bbox = bboxes[i].item()
    target = file[bbox]['label']
    if len(target) > 1:
        labels.append([file[target.value[j].item()].value[0][0] for j in range(len(target))])
    else:
        labels.append([target.value[0][0]])
print('asd')