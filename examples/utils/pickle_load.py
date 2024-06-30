import pickle


def pkl_load(fname):
    content = pickle.load(open(fname, "rb"))
    return content


if __name__ == "__main__":
    c = pkl_load("../tests/data/BraTS21_base_ep1k.model.pkl")
    print(c.keys())
    print(c['plans'].keys())
    print(c['plans']['plans_per_stage'][0]['pool_op_kernel_sizes'])
