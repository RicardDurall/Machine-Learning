import CNN, csv


weights = ["/home/ricard/Desktop/Kaggle/Plants/weights0.hdf5",
           "/home/ricard/Desktop/Kaggle/Plants/weights1.hdf5",
           "/home/ricard/Desktop/Kaggle/Plants/weights2.hdf5",
           "/home/ricard/Desktop/Kaggle/Plants/weights3.hdf5",
           "/home/ricard/Desktop/Kaggle/Plants/weights4.hdf5"]

weights = ["/home/ricard/Desktop/Kaggle/Plants/weightsX0.hdf5"]

rows = CNN.prediction(weights)


with open("/home/ricard/Desktop/Kaggle/Plants/output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "species"])
    for row in rows:
        writer.writerow(row)
