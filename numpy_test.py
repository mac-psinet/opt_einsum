from __future__ import division, absolute_import, print_function

import os
import sys
import itertools
import traceback

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_array_equal,
    assert_almost_equal, assert_allclose, run_module_suite,
    dec
)

from opt_einsum import contract
import time

### Build dictionary of tests

class TestContract(object):
    def setup(self, n=1):
        chars = 'abcdefghij'
        sizes = np.array([2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3]) 
        if n!=1:
            sizes *= 1 + np.random.rand(sizes.shape[0]) * n 
            sizes = sizes.astype(np.int)
        self.sizes = {c: s for c, s in zip(chars, sizes)}

    def compare(self, string, *views):
        if len(views)==0:
            views = []
            terms = string.split('->')[0].split(',')
            for term in terms:
                dims = [self.sizes[x] for x in term]
                views.append(np.random.rand(*dims))

        ein = np.einsum(string, *views)
        opt = contract(string, *views)
        assert_allclose(ein, opt)

    def test_hadamard_like_products(self):
        self.compare('a,ab,abc->abc')
        self.compare('a,b,ab->ab')

    def test_index_transformations():
        self.compare('ea,fb,gc,hd,abcd->efgh')
        self.compare('ea,fb,abcd,gc,hd->efgh')
        self.compare('abcd,ea,fb,gc,hd->efgh')
    
    def test_complex(self):
        self.compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.compare('acdf,jbje,gihb,hfac,gfac,gifabc,hfac')
        self.compare('cd,bdhe,aidb,hgca,gc,hgibcd,hgac')
        self.compare('abhe,hidj,jgba,hiab,gab')
        self.compare('bde,cdh,agdb,hica,ibd,hgicd,hiac')
        self.compare('chd,bde,agbc,hiad,hgc,hgi,hiad')
        self.compare('chd,bde,agbc,hiad,bdi,cgh,agdb')
        self.compare('bdhe,acad,hiab,agac,hibd')

    def test_collapse(self):
        self.compare('ab,ab,c->')
        self.compare('ab,ab,c->c')
        self.compare('ab,ab,cd,cd->')
        self.compare('ab,ab,cd,cd->ac')
        self.compare('ab,ab,cd,cd->cd')
        self.compare('ab,ab,cd,cd,ef,ef->')

    def test_expand(self):
        self.compare('ab,cd,ef->abcdef')
        self.compare('ab,cd,ef->acdf')
        self.compare('ab,cd,de->abcde')
        self.compare('ab,cd,de->be')
        self.compare('ab,bcd,cd->abcd')
        self.compare('ab,bcd,cd->abd') 

    def test_previously_failed(self):
        self.compare('eb,cb,fb->cef')
        self.compare('dd,fb,be,cdb->cef')
        self.compare('bca,cdb,dbf,afc->')
        self.compare('dcc,fce,ea,dbf->ab')
        self.compare('fdf,cdd,ccd,afe->ae')
        self.compare('abcd,ad')
        self.compare('ed,fcd,ff,bcf->be')
        self.compare('baa,dcf,af,cde->be')
        self.compare('bd,db,eac->ace')

    def test_inner_product(self): 
        # Inner products
        self.compare('ab,ab')
        self.compare('ab,ba')
        self.compare('abc,abc')
        self.compare('abc,bac')
        self.compare('abc,cba')

    def test_dot_product(self):
        # GEMM test cases
        self.compare('ab,bc')
        self.compare('ab,cb')
        self.compare('ba,bc')
        self.compare('ba,cb')
        self.compare('abcd,cd')
        self.compare('abcd,ab')
        self.compare('abcd,cdef')
        self.compare('abcd,cdef->feba')
        self.compare('abcd,efdc')
        # Inner than dot
        self.compare('aab,bc->ac')
        self.compare('ab,bcc->ac')
        self.compare('aab,bcc->ac')
        self.compare('baa,bcc->ac')
        self.compare('aab,ccb->ac')

    def test_random_cases(self):
        # Randomly build test caes
        self.compare('aab,fa,df,ecc->bde')
        self.compare('ecb,fef,bad,ed->ac')
        self.compare('bcf,bbb,fbf,fc->')
        self.compare('bb,ff,be->e')
        self.compare('bcb,bb,fc,fff->')
        self.compare('fbb,dfd,fc,fc->')
        self.compare('afd,ba,cc,dc->bf')
        self.compare('adb,bc,fa,cfc->d')
        self.compare('bbd,bda,fc,db->acf')
        self.compare('dba,ead,cad->bce')
        self.compare('aef,fbc,dca->bde')

    def test_einsum_errors(self):
        # Need enough arguments
        # assert_raises(ValueError, contract)
        # assert_raises(ValueError, contract, "")

        # subscripts must be a string
        assert_raises(TypeError, contract, 0, 0)

        # order parameter must be a valid order
        # assert_raises(TypeError, contract, "", 0, order='W')

        # casting parameter must be a valid casting
        # assert_raises(ValueError, contract, "", 0, casting='blah')

        # dtype parameter must be a valid dtype
        # assert_raises(TypeError, contract, "", 0, dtype='bad_data_type')

        # other keyword arguments are rejected
        # assert_raises(TypeError, contract, "", 0, bad_arg=0)

        # issue 4528 revealed a segfault with this call
        # assert_raises(TypeError, contract, *(None,)*63)

        # number of operands must match count in subscripts string
        assert_raises(ValueError, contract, "", 0, 0)
        assert_raises(ValueError, contract, ",", 0, [0], [0])
        assert_raises(ValueError, contract, ",", [0])

        # can't have more subscripts than dimensions in the operand
        assert_raises(ValueError, contract, "i", 0)
        assert_raises(ValueError, contract, "ij", [0, 0])
        assert_raises(ValueError, contract, "...i", 0)
        assert_raises(ValueError, contract, "i...j", [0, 0])
        assert_raises(ValueError, contract, "i...", 0)
        assert_raises(ValueError, contract, "ij...", [0, 0])

        # invalid ellipsis
        assert_raises(ValueError, contract, "i..", [0, 0])
        assert_raises(ValueError, contract, ".i...", [0, 0])
        assert_raises(ValueError, contract, "j->..j", [0, 0])
        assert_raises(ValueError, contract, "j->.j...", [0, 0])

        # invalid subscript character
        assert_raises(ValueError, contract, "i%...", [0, 0])
        assert_raises(ValueError, contract, "...j$", [0, 0])
        assert_raises(ValueError, contract, "i->&", [0, 0])

        # output subscripts must appear in input
        assert_raises(ValueError, contract, "i->ij", [0, 0])

        # output subscripts may only be specified once
        assert_raises(ValueError, contract, "ij->jij", [[0, 0], [0, 0]])

        # dimensions much match when being collapsed
        assert_raises(ValueError, contract, "ii", np.arange(6).reshape(2, 3))
        assert_raises(ValueError, contract, "ii->i", np.arange(6).reshape(2, 3))

        # broadcasting to new dimensions must be enabled explicitly
        assert_raises(ValueError, contract, "i", np.arange(6).reshape(2, 3))
        assert_raises(ValueError, contract, "i->i", [[0, 1], [0, 1]],
                                            out=np.arange(4).reshape(2, 2))

    def test_einsum_broadcast(self):
        # Issue #2455 change in handling ellipsis
        # remove the 'middle broadcast' error
        # only use the 'RIGHT' iteration in prepare_op_axes
        # adds auto broadcast on left where it belongs
        # broadcast on right has to be explicit

        A = np.arange(2*3*4).reshape(2,3,4)
        B = np.arange(3)
        ref = np.einsum('ijk,j->ijk',A, B)
        self.compare('ij...,j...->ij...', A, B)
        self.compare('ij...,...j->ij...', A, B)
        self.compare('ij...,j->ij...', A, B) # used to raise error

        A = np.arange(12).reshape((4,3))
        B = np.arange(6).reshape((3,2))
        ref = np.einsum('ik,kj->ij',A, B)
        self.compare('ik...,k...->i...', A, B)
        self.compare('ik...,...kj->i...j', A, B)
        self.compare('...k,kj', A, B) # used to raise error
        self.compare('ik,k...->i...', A, B) # used to raise error

        dims=[2,3,4,5];
        a = np.arange(np.prod(dims)).reshape(dims)
        v = np.arange(dims[2])
        ref = np.einsum('ijkl,k->ijl', a, v)
        self.compare('ijkl,k', a, v)
        self.compare('...kl,k', a, v)  # used to raise error
        self.compare('...kl,k...', a, v)
        # no real diff from 1st

        J,K,M=160,160,120;
        A=np.arange(J*K*M).reshape(1,1,1,J,K,M)
        B=np.arange(J*K*M*3).reshape(J,K,M,3)
        ref = np.einsum('...lmn,...lmno->...o', A, B)
        self.compare('...lmn,lmno->...o', A, B)  # used to raise error

t = time.time()
c = TestContract()
c.setup()
c.test_hadamard_like_products()
c.test_complex()
c.test_collapse()
c.test_expand()
c.test_expand()
c.test_previously_failed()
c.test_inner_product()
c.test_dot_product()
c.test_random_cases()
#c.test_einsum_errors()
c.test_einsum_broadcast()
print(time.time()-t)

