import copy
import numpy as np
import numpy.random as rnd
import pyopencl as cl
import random
import time
from scipy.stats.stats import pearsonr

rnd.seed()

def main():
    # Config
    steps = 40000
    num_gbl = 2048
    num_lcl = 128
    num_grp = num_gbl / num_lcl
    num_sec = 25
    num_keep = 2

    # Generate some input
    alpha = rnd.uniform(-1,1,size=num_sec).astype(np.float32)
    prc = rnd.uniform(1,10, size=num_sec).astype(np.float32) * 10
    bid = prc - np.multiply(rnd.uniform(size=prc.size), prc/100).astype(np.float32)
    ask = prc + np.multiply(rnd.uniform(size=prc.size), prc/100).astype(np.float32)
    adv = rnd.uniform(10000000, size=num_sec).astype(np.float32)
    port_out = np.zeros((num_grp,num_sec), dtype=np.int32)

    def get_fit(p):
        s = np.float64(0)
        for i in range(len(p)):
            s += alpha[i] * p[i]
        return s

    def get_max(res):
        m = None
        f = 0
        for fit, port in map((lambda p: (get_fit(p), p)), res):
            gmv = sum(abs(port[i]*prc[i]) for i in range(len(port)))
            if fit * gmv > f or m is None:
                f = fit * gmv
                m = port
        return f, m

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    
    # Create OpenCL buffers
    mf = cl.mem_flags
    sec_rnd = rnd.uniform(size=num_sec).astype(np.float32)
    thd_rnd = rnd.uniform(size=num_gbl).astype(np.float32)
    alpha_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=alpha)
    bid_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bid)
    ask_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ask)
    prc_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=prc)
    adv_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=adv)
    port_buf = cl.Buffer(ctx, mf.WRITE_ONLY, port_out.nbytes)
    thd_rnd_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thd_rnd)
    sec_rnd_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sec_rnd)
    port_scratch_buf = cl.Buffer(ctx, mf.READ_WRITE, num_gbl * num_sec * 4)
    fit_buf = cl.Buffer(ctx, mf.READ_WRITE, num_gbl * 8)
    keep_buf = cl.LocalMemory(num_keep * 4)


    prg = cl.Program(ctx, open('kernel.c').read()).build()

    # Init buffers
    e = prg.init(queue, (num_gbl,), (num_lcl,),
        port_scratch_buf,
        thd_rnd_buf,
        fit_buf,
        np.int32(num_sec),
        np.int32(num_gbl),
        np.int32(rnd.randint(0,max(num_sec,num_gbl))))
    e = cl.enqueue_barrier(queue, wait_for=[e])

    # Init fitness
    e = prg.get_fitness(queue, (num_gbl,), (num_lcl,),
        alpha_buf,
        bid_buf,
        ask_buf,
        prc_buf,
        adv_buf,
        port_scratch_buf,
        fit_buf,
        thd_rnd_buf,
        np.int32(num_sec),
        np.int32(num_gbl), wait_for=[e])
    e = cl.enqueue_barrier(queue, wait_for=[e])

    for i in range(0, steps - 1):
        sec_rnd = rnd.uniform(size=num_sec).astype(np.float32)
        thd_rnd = rnd.uniform(size=num_gbl).astype(np.float32)
        thd_rnd_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=thd_rnd)
        sec_rnd_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sec_rnd)

        # Reap, mutate, and recombinate
        e = prg.mutate(queue, (num_gbl,), (num_lcl,),
            port_scratch_buf,
            fit_buf,
            sec_rnd_buf,
            thd_rnd_buf,
            keep_buf,
            np.int32(num_sec),
            np.int32(num_gbl),
            np.int32(num_keep),
            np.int32(rnd.randint(0,max(num_sec,num_gbl))), wait_for=[e])
        e = cl.enqueue_barrier(queue, wait_for=[e])

        # Recomute fitness
        e = prg.get_fitness(queue, (num_gbl,), (num_lcl,),
            alpha_buf,
            bid_buf,
            ask_buf,
            prc_buf,
            adv_buf,
            port_scratch_buf,
            fit_buf,
            thd_rnd_buf,
            np.int32(num_sec),
            np.int32(num_gbl), wait_for=[e])
        e = cl.enqueue_barrier(queue, wait_for=[e])

    # Get the top portfolios from each work group
    port_buf = cl.Buffer(ctx, mf.WRITE_ONLY, port_out.nbytes)
    e = prg.get_max(queue, (num_gbl,), (num_lcl,),
        port_scratch_buf,
        port_buf,
        fit_buf,
        np.int32(num_sec), wait_for=[e])
    e = cl.enqueue_barrier(queue, wait_for=[e])
    e = cl.enqueue_copy(queue, port_out, port_buf, wait_for=[e])
    e = cl.enqueue_barrier(queue, wait_for=[e])
    
    # (Meta-)Select the one we want
    f, port_out = get_max(port_out)
    
    print('Signal:')
    print(alpha)
    print('Prices:')
    print(prc)
    print('ADV:')
    print(adv)
    print('Spreads:')
    print(ask - bid)
    print('Portfolio:')
    print(port_out)
    print('Fitness:', f)
    print('Max Participation:', max(abs(port_out[i]*prc[i]/adv[i]) for i in range(len(port_out))))
    print('GMV:', sum(abs(port_out[i]*prc[i]) for i in range(len(port_out))))
    print('NMV:', sum(port_out[i]*prc[i] for i in range(len(port_out))))
    print('Peason R bt alpha and port:', pearsonr(port_out, alpha))


if __name__ == '__main__':
   main()

