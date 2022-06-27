Extract the images:

    mkdir images && tar -xf family-faces.tar.gz -C images/

Use `family.py` to load the data:

    import family as f
    Xs, ys, ts = f.family_dataset('images/')

The dataset is based on images from A. Gallagher and T. Chen, “Understanding
groups of images of people,” in IEEE Conference on Computer Vision and Pattern
Recognition, 2009, pp. 256–263.
