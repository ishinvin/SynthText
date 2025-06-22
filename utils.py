import h5py

def add_res_to_db(imgname, res, db):
    """
    Add the synthetically generated text image instance
    and other metadata to the dataset.
    """
    ninstance = len(res)
    for i in range(ninstance):
        dname = "%s_%d"%(imgname, i)
        db['data'].create_dataset(dname,data=res[i]['img'])
        db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
        db['data'][dname].attrs.create('txt', res[i]['txt'], dtype=h5py.string_dtype(encoding='utf-8'))