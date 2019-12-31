import pyscf
from pyscf import scf
from pyscf import gto
from pyscf.mp.mp2 import *
import numpy as np
from pyscf import lib
mol = gto.Mole()
mol.atom = [
    [8 , (0. , 0.    , 0.)],
    [1 , (0. , -0.757, 0.587)],
    [1 , (0. , 0.757 , 0.587)]]

mol.basis = 'cc-pvdz'
mol.build()
mf = scf.RHF(mol).run()

pt = MP2(mf)

emp2, t2 = pt.kernel()
nocc = pt.nocc
nmo = pt.nmo
nvir = nmo-nocc
mo_coeff=pt.mo_coeff
mo_energy=pt.mo_energy
eris=pt.ao2mo(mo_coeff)
"""pyscfもともとのやつ
eia = pt.mo_energy[:nocc,None] - pt.mo_energy[None,nocc:]
emp2=0
for i in range(nocc):
    gi = eris.ovov[i*nvir:(i+1)*nvir]

    gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)

    t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])


    emp2 += np.einsum('jab,jab',t2i,gi) *2
    emp2 -= np.einsum('jab,jba',t2i,gi) 
print(emp2)
"""
#以下、愚直にijabの足し合わせ
emp2=0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                e=mo_energy[i]+mo_energy[j]-mo_energy[nocc+a]-mo_energy[nocc+b]
                
                abij=eris.ovov[nvir*i+a][nvir*j+b]
                abji=eris.ovov[nvir*j+a][nvir*i+b]
                
                emp2+= (2*(abij.conj()*abij)-(abij.conj()*abji))/e
print(emp2)

