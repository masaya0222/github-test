import numpy
from pyscf import gto, ao2mo
import h5py
def view(h5file, dataname='eri_mo'):
    with h5py.File(h5file, 'r') as f5:
        print('dataset %s, shape %s ' % (str(f5.keys()), str(f5[dataname].shape)))
mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='sto3g')
mo1 = numpy.random.random((mol.nao_nr(),10))

eri1 = ao2mo.full(mol, mo1)
print(eri1.shape)

eri = mol.intor('int2e_sph', aosym='s8')
eri1 = ao2mo.full(eri, mo1, compact=False)
print(eri1.shape)
