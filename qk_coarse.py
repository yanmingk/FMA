import torch
import os.path
from mla.configs import SEARCHS, MASKVAL, TUNE_VERBOSE, CUDA_MODEL, CUDA_ARCH
import tvm
from tvm import te , auto_scheduler

@auto_scheduler.register_workload
def qk_coarse_ltr_workload(b, n, d, nl, w, p, dtype):
    X = te.placeholder((b, n, d), name='X', dtype=dtype)
    Y = te.placeholder((b, nl+ 2*p, d), name='Y', dtype=dtype)
    k = te.reduce_axis((0, d), name='k')
    output_shape = (b, n, 2*p)
    maskval = MASKVAL / (d+1)
    algorithm = lambda l, i, j: te.sum(
        te.if_then_else(
            te.floordiv(te.floormod(i,2*w),w) >= te.floordiv(j,p),
            X[l, i, k] * Y[l, 2*p* te.floordiv(i,2*w) + j, k],
            maskval
        ),
        axis = k,
    )
    Z = te.compute(output_shape, algorithm, name='Z')
    return [X, Y, Z]

@auto_scheduler.register_workload
def qk_coarse_ltr_backx_workload(b, n, d, nl, w, p, dtype):
    grad_output = te.placeholder((b, n, 2*p), name='grad_output', dtype=dtype)
    t2 = te.placeholder((b, nl+2*p, d), name='t2', dtype=dtype)
    k = te.reduce_axis((0, 2*p), name='k')
    output_shape = (b, n, d)
    algorithm_x = lambda l,i,j: te.sum(
        grad_output[l,i,k] * t2[l,2*p* te.floordiv(i,2*w) + k, j],
        axis=k,
        where = te.floordiv(te.floormod(i,2*w),w) >= te.floordiv(k,p)
    ) 
    grad_t1 = te.compute(output_shape, algorithm_x, name='grad_t1')
    return [t2, grad_output, grad_t1]

@auto_scheduler.register_workload
def qk_coarse_ltr_backy_workload(b, n, d, nl, w, p, dtype):
    grad_output = te.placeholder((b, n + 2*w, 2*p), name='grad_output', dtype=dtype)
    t1 = te.placeholder((b, n + 2*w, d), name='t1', dtype=dtype)
    k = te.reduce_axis((0, 2*w), name='k')
    output_shape = (b, nl, d)

    algorithm_y = lambda l,i,j: te.sum(
            grad_output[l, 2*w*(1+te.floordiv(i,2*p))+ k, te.floormod(i,2*p)] * 
                     t1[l, 2*w*(1+te.floordiv(i,2*p))+ k, j],
            axis = k,
            where= te.floordiv(k, w) >= te.floordiv(te.floormod(i,2*p), p)
        )
    grad_t2 = te.compute(output_shape, algorithm_y, name='grad_t2')
    return [t1, grad_output, grad_t2]

class DiagonaledMMQKCoarse(torch.autograd.Function):
    function_dict = {}
    @staticmethod
    def _compile_function(dtype: str, device: str, b,n,d,nl,p):
        print("compiling qk_coarse forward for",b,n,d,nl,p,dtype)
        w = p*n//nl
        target = tvm.target.cuda(model=CUDA_MODEL, arch=CUDA_ARCH)
        task = auto_scheduler.SearchTask(
            func=qk_coarse_ltr_workload, args=(b, n, d, nl, w, p, dtype), target=target
        )
        print("Computational DAG:")
        print(task.compute_dag)
        log_file = "aslogs/qk_coarse_ltr.json"
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=SEARCHS,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=TUNE_VERBOSE,
        )
        task.tune(tune_option)
        sch, args = task.apply_best(log_file)
        del measure_ctx
        print("Lowered TIR:")
        print(tvm.lower(sch, args, simple_mode=True))
        func = tvm.build(sch, args, target)
        return func

    @staticmethod
    def _compile_function_back_x(dtype: str, device: str, b,n,d,nl,p):
        print("compiling qk_coarse backx for",b,n,d,nl,p,dtype)
        target = tvm.target.cuda(model=CUDA_MODEL, arch=CUDA_ARCH)
        w = p*n//nl
        taskx = auto_scheduler.SearchTask(
            func=qk_coarse_ltr_backx_workload, args=(b, n, d, nl, w, p, dtype), target=target
        )
        print("Computational DAG:")
        print(taskx.compute_dag)
        log_file = "aslogs/qk_coarse_ltr_backx.json"
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=SEARCHS,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=TUNE_VERBOSE,
        )
        taskx.tune(tune_option)
        schx, argsx = taskx.apply_best(log_file)
        del measure_ctx
        print("Lowered TIR:")
        print(tvm.lower(schx, argsx, simple_mode=True))
        funcx = tvm.build(schx, argsx, target)
        return funcx

    @staticmethod
    def _compile_function_back_y(dtype: str, device: str, b,n,d,nl,p):
        print("compiling qk_coarse backy for",b,n,d,nl,p,dtype)
        w = p*n//nl
        target = tvm.target.cuda(model=CUDA_MODEL, arch=CUDA_ARCH)
        tasky = auto_scheduler.SearchTask(
            func=qk_coarse_ltr_backy_workload, args=(b, n, d, nl, w, p, dtype), target=target
        )
        print("Computational DAG:")
        print(tasky.compute_dag)
        log_file = "aslogs/qk_coarse_ltr_backy.json"
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=SEARCHS,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
            verbose=TUNE_VERBOSE,
        )
        tasky.tune(tune_option)
        schy, argsy = tasky.apply_best(log_file)
        del measure_ctx
        print("Lowered TIR:")
        print(tvm.lower(schy, argsy, simple_mode=True))
        funcy = tvm.build(schy, argsy, target)
        return funcy

    @staticmethod
    def _get_lib_filename(dtype: str, device: str, functype: str, b,n,d,nl,p):
        base_filename = 'funcs/qk_coarse'
        return '{}_{}_{}_{}_{}_b{}_n{}_d{}_nl{}_p{}.so'.format(
            base_filename, CUDA_MODEL, dtype, device, functype,
            b,n,d,nl,p
        )

    @staticmethod
    def _save_compiled_function(f, dtype: str, device: str, functype: str,b,n,d,nl,p):
        if not os.path.exists('funcs/'):
            os.makedirs('funcs/')
        f.export_library(DiagonaledMMQKCoarse._get_lib_filename(dtype, device, functype,b,n,d,nl,p))

    @staticmethod
    def _load_compiled_function(dtype: str, device: str, functype: str,b,n,d,nl,p):
        from tvm.runtime import load_module
        filename = DiagonaledMMQKCoarse._get_lib_filename(dtype, device, functype,b,n,d,nl,p)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_dirs = ['../../', '../', './', f'{current_dir}/', f'{current_dir}/../']
        for potential_dir in  potential_dirs:
            filepath = '{}{}'.format(potential_dir, filename)
            if os.path.isfile(filepath):
                print('Loading tvm binary from: {}'.format(filepath))
                return load_module(filepath)
        return None

    @staticmethod
    def _get_function(dtype: str, device: str, functype: str,b,n,d,nl,p):
        args = (dtype, device, functype, b,n,d,nl,p)
        if args not in DiagonaledMMQKCoarse.function_dict:
            f = DiagonaledMMQKCoarse._load_compiled_function(dtype, device, functype,b,n,d,nl,p)
            if not f:
                print('Tvm binary not found. Compiling ...' + functype)
                if functype == 'forward':
                    f = DiagonaledMMQKCoarse._compile_function(dtype, device,b,n,d,nl,p)
                elif functype == 'backx':
                    f = DiagonaledMMQKCoarse._compile_function_back_x(dtype, device,b,n,d,nl,p)
                elif functype == 'backy':
                    f = DiagonaledMMQKCoarse._compile_function_back_y(dtype, device,b,n,d,nl,p)
                DiagonaledMMQKCoarse._save_compiled_function(f, dtype, device, functype,b,n,d,nl,p)
            from tvm.contrib import dlpack
            f_pytorch = dlpack.to_pytorch_func(f)
            DiagonaledMMQKCoarse.function_dict[args] = f_pytorch
        return DiagonaledMMQKCoarse.function_dict[args]

    @staticmethod
    def _diagonaled_mm(t1: torch.Tensor, t2: torch.Tensor,p):
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        b = t1.shape[0]
        n = t1.shape[1]
        d = t1.shape[2]
        nl = t2.shape[1]
        t2_pad = torch.cat(
            (torch.zeros((b,2*p,d), dtype=t2.dtype, device=t2.device), t2),
            dim=1
        )
        Z = torch.zeros((b, n, 2*p), dtype=t1.dtype, device=t1.device)
        _diagonaled_mm_function = DiagonaledMMQKCoarse._get_function(dtype, device, 'forward',b,n,d,nl,p)
        _diagonaled_mm_function(t1, t2_pad, Z)
        return Z

    @staticmethod
    def _back_x(t2: torch.Tensor, grad_output: torch.Tensor,p):
        dtype = str(t2.dtype).split('.')[1]
        device = t2.device.type
        b = t2.shape[0]
        nl = t2.shape[1]
        d = t2.shape[2]
        n = grad_output.shape[1] 

        t2_pad = torch.cat(
            (torch.zeros((b,2*p,d), dtype=t2.dtype, device=t2.device),
                t2,),
            dim=1
        )
        grad_t1 = t2.new_empty(b,n,d)
        _back_x_function = DiagonaledMMQKCoarse._get_function(dtype, device, 'backx',b,n,d,nl,p)
        _back_x_function(t2_pad, grad_output, grad_t1)
        return grad_t1

    @staticmethod
    def _back_y(t1: torch.Tensor, grad_output: torch.Tensor, nl,p):
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        b = t1.shape[0]
        n = t1.shape[1]
        d = t1.shape[2]
        w = p*n//nl
        grad_t2_A = t1.new_empty(b,nl,d)
        grad_output_pad = torch.cat(
            [   grad_output,
                torch.zeros((b,2*w,2*p), dtype=grad_output.dtype, device=grad_output.device)],
            dim=1
        )
        t1_pad = torch.cat(
            [   t1,\
                torch.zeros((b,2*w,d), dtype=t1.dtype, device=t1.device)],
            dim=1
        )
        _back_ya_function = DiagonaledMMQKCoarse._get_function(dtype, device, 'backy',b,n,d,nl,p)
        _back_ya_function(t1_pad, grad_output_pad, grad_t2_A)
        return grad_t2_A

    @staticmethod
    def _prepare_tensors(t):
        assert t.is_contiguous()
        t_stride = list(t.stride())
        t_size = list(t.size())
        if t_size[0] == 1 and t_stride[0] == t_stride[1]:
            t_stride[0] = t_size[1] * t_size[2] * t_size[3]
            t = t.as_strided(size=t_size, stride=t_stride)
        return t

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor, p: int):
        ctx.save_for_backward(t1, t2)
        ctx.nl = t2.shape[1]
        ctx.p = p
        t1 = DiagonaledMMQKCoarse._prepare_tensors(t1)
        t2 = DiagonaledMMQKCoarse._prepare_tensors(t2)
        output = DiagonaledMMQKCoarse._diagonaled_mm(t1, t2,p)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        t1, t2 = ctx.saved_tensors
        nl, p = ctx.nl, ctx.p
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_output = DiagonaledMMQKCoarse._prepare_tensors(grad_output)
        t1 = DiagonaledMMQKCoarse._prepare_tensors(t1)
        t2 = DiagonaledMMQKCoarse._prepare_tensors(t2)
        grad_t1 = DiagonaledMMQKCoarse._back_x(t2, grad_output,p)
        grad_t2 = DiagonaledMMQKCoarse._back_y(t1, grad_output, nl,p)
        return grad_t1, grad_t2, None

diagonaled_mm_qk_coarse_ltr = DiagonaledMMQKCoarse.apply