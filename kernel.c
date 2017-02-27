#pragma OPENCL EXTENSION cl_khr_fp64 : enable

int count_larger(__global double* fit) {
    const int gid = get_global_id(0);
    const int grpid = get_group_id(0);
    const int num_lcl = get_local_size(0);
    const int lid = get_local_id(0);

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    int num_larger = 0;
    for (int i = grpid * num_lcl; i < grpid * num_lcl + num_lcl; i++) {
        if (fit[i] > fit[gid] || fit[gid] < 1) {
            num_larger++;
        }
    }
    return num_larger;
}

__kernel
void init(
    __global int* _port,
    __global float* thd_rnd,
    __global float* fit,
    const int num_sec,
    const int num_thd,
    const int rnd_off)
{
    const int gid = get_global_id(0);
    __global int* port = _port + gid * num_sec;
    for (int i = 0; i < num_sec; i++)
        port[i] = 10 * thd_rnd[(gid + rnd_off + i) % num_thd] - 5;
    fit[gid] = 0;
}

__kernel
void get_fitness(
	__constant float* alpha,
	__constant float* bid,
	__constant float* ask,
	__constant float* prc,
	__constant float* adv,
	__global int* _port_s,
	__global double* fit,
    __global float* thd_rnd,
	const int num_sec,
    const int num_thd)
{
	const int gid = get_global_id(0);

	__global int* port = _port_s + gid * num_sec;
    const double target_gmv = 200000;
    const double max_nmv = 0.025 * target_gmv;
    const double max_participation = .025;
    double fit_ = 0;
    float gmv = 0;
    float nmv = 0;
    float max_part = 0;
	for (int i = 0; i < num_sec; i++) {
        fit_ += alpha[i] * port[i] * adv[i] / fabs(ask[i]-bid[i]);
        gmv += fabs(port[i]*prc[i]);
		nmv += port[i]*prc[i];
        max_part = max(max_part, fabs(port[i] * prc[i] / adv[i]));
	}
    fit_ *= gmv;
    fit_ *= max_participation / max_part;

    if (!(gmv < target_gmv) || 
            !(fabs(nmv) < max_nmv) || 
            !(max_part < max_participation) ||
            !(fit_ > 0))
    {
        fit_ = -1.0;
        for (int i = 0; i < num_sec; i++) {
            port[i] /= 10;
        }
    }
    fit[gid] = fit_;
}

__constant float mut_prob = .3;
__constant float recom_prob = .5;
__kernel
void mutate(
    __global int* _port,
    __global double* fit,
    __global float* sec_rnd,
    __global float* thd_rnd,
    __local int* best,
    const int num_sec,
    const int num_thd,
    const int num_keep,
    const int rnd_off)
{
    int gid = get_global_id(0);
    int larger = count_larger(fit);
    bool keep = larger == 0;
    bool mutate = thd_rnd[(rnd_off + gid) % num_thd] < mut_prob;
    bool recom = !mutate && thd_rnd[(rnd_off + gid) % num_thd] < recom_prob;
    __global int* port = _port + gid * num_sec;

    for (int i = 0; i < num_keep; i++) best[i] = -1;

    if (mutate && !keep) {
        double avg = 0;
        for (int i = 0; i < num_sec; i++) {
            port[i] = (int)(port[i] * (4 * sec_rnd[(i + rnd_off + gid) % num_sec] - 2) + 5*sec_rnd[(i + rnd_off*i + gid) % num_sec]);
            port[i] *= sec_rnd[(i + rnd_off + gid) % num_sec] < .2 ? -1 : 1;
            avg = (avg * i + abs(port[i])) / (i + 1);
        }
    }

    if (!mutate && !keep) {
        for (int i = 0; i < num_sec; i++) {
            port[i] = (int)(5 * sec_rnd[(i + rnd_off + gid) % num_sec] - 2.5);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (keep) {
        best[larger] = gid;
    }

    if (recom && !keep) {
        int recom_idx = best[convert_int(thd_rnd[(rnd_off + gid) % num_thd] * num_keep) % num_keep];
        __global int* best_port = _port + recom_idx * num_sec; 
        if (recom_idx == -1) return;
        for (int i = 0; i < num_sec; i++) {
            port[i] = (port[i] + best_port[i]) / (2 * thd_rnd[(rnd_off + gid) % num_thd]);
        }
    }
}

__kernel
void get_max(
	__global int* _port,
    __global int* _out_port,
	__global double* fit,
	const int num_sec)
{
	int gid = get_global_id(0);
    int grpid = get_group_id(0);
    bool best = count_larger(fit) == 0;

    __global int* port = _port + gid * num_sec; 
    __global int* out_port = _out_port + grpid * num_sec;
    if (best) {
        for (int i = 0; i < num_sec; i++) {
            out_port[i] = port[i];
        }
    }
}

