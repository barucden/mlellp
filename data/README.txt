Extract the images:

    mkdir images && tar -xf family-faces.tar.gz -C images/

Use `family.py` to load the data:

    import family as f
    Xs, ys, ts = f.family_dataset('images/')

