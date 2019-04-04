# Standard scientific Python Imports
import imageio as imageio
import matplotlib.pyplot as plt
from sklearn import datasets, svm
# Import datasets, classifiers and performance metrics
from scipy.misc import imresize, bytescale
# The digits datasets
digits = datasets.load_digits()
print ('digits:', digits.keys())
print ("digits.target-----:", digits.target)
images_and_labels = list(zip(digits.images, digits.target))
print ("len(images_and_labels)", len(images_and_labels))

for index, [image, label] in enumerate(images_and_labels[:5]):
    print("index:", index, "image :\n", image, "label:", label)
    plt.subplot(2, 5, index + 1)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training %i' % label)
n_samples = len(digits.images)
print ("n_sample", n_samples)
imageData = digits.images.reshape((n_samples, -1))
print ("After Reshaped: len(imageData[0])", len(imageData[0]))
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
# We learn the digits on the first half of the digit
classifier.fit(imageData[:n_samples // 2], digits.target[: n_samples // 2])
# Now predict the value of the digits on the second half
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(imageData[n_samples // 2:])

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, [image, predictions] in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('prediction:%i' % predictions)
print ("original values", digits.target[n_samples // 2:(n_samples // 2) + 5])
plt.show()
# Install Pillow library
img = imageio.imread("Seven.jpeg")
img = imresize(img, (8, 8))
img = img.astype(digits.images.dtype)

img = bytescale(img, high=16.0)
print ("img:", img)
x_testData = []

for c in img:
    for r in c:
        x_testData.append(sum(r) / 3.0)
print ("x_testData:", x_testData)

x_testData = [x_testData]
print ("len(x_testData):", len(x_testData))
print ("Machine output", classifier.predict(x_testData))
plt.show()
