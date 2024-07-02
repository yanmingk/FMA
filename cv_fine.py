import torch
import os.path
from mla.configs import SEARCHS, TUNE_VERBOSE, CUDA_MODEL, CUDA_ARCH

class DiagonaledMMCVFine(torch.autograd.Function):
    function_dict = {}

    @staticmethod
    def _compile_function(dtype: str, device: str, b,n,d,m):
        print("compiling cv_fine forward for",b,n,d,m,dtype)
        import tvm
        from tvm import te , auto_scheduler
        @auto_scheduler.register_workload
        def cv_fine_ltr_workload(b, n, d, m):
            X = te.placeholder((b, n, 2*m), name='X', dtype=dtype)
            Y = te.placeholder((b, n+ m, d), name='Y', dtype=dtype)
            k = te.reduce_axis((0, 2*m), name='k')
            output_shape = (b, n, d)
            algorithm = lambda l, i, j: te.sum( 
                X[l, i, k] * Y[l, m * te.floordiv(i,m) +k, j],
                axis = k,
                where = (k <= te.floormod(i,m) + m)
            )
            Z = te.compute(output_shape, algorithm, name='Z')
            return [X, Y, Z]
        target = tvm.target.cuda(model=CUDA_MODEL, arch=CUDA_ARCH)
        task = auto_scheduler.SearchTask(
            func=cv_fine_ltr_workload, args=(b, n, d, m), target=target
        )
        print("Computational DAG:")
        print(task.compute_dag)
        log_file = "aslogs/cv_fine_ltr.json"
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
    def _compile_function_back_x(dtype: str, device: str, b,n,d,m):
        print("compiling cv_fine backx for",b,n,d,m,dtype)
        import tvm
        from tvm import te , auto_scheduler
        @auto_scheduler.register_workload
        def cv_fine_ltr_backx_workload(b, n, d, m):
            grad_output = te.placeholder((b, n, d), name='grad_output', dtype=dtype)
            t2 = te.placeholder((b, n+ m, d), name='t2', dtype=dtype)
            k = te.reduce_axis((0, d), name='k')
            output_shape = (b, n, 2*m)
            algorithm_x = lambda l,i,j: te.sum(
                grad_output[l, i, k] * t2[l, m* te.floordiv(i,m)+ j, k],
                axis = k,
                where = (te.floormod(i,m)+m >= j)
            )
            grad_t1 = te.compute(output_shape, algorithm_x, name='grad_t1')
            return [t2, grad_output, grad_t1]
        target = tvm.target.cuda(model=CUDA_MODEL, arch=CUDA_ARCH)
        taskx = auto_scheduler.SearchTask(
            func=cv_fine_ltr_backx_workload, args=(b, n, d, m), target=target
        )
        print("Computational DAG:")
        print(taskx.compute_dag)
        log_file = "aslogs/cv_fine_ltr_backx.json"
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
    def _compile_function_back_y(dtype: str, device: str, b,n,d,m):
        print("compiling cv_fine backy for",b,n,d,m,dtype)
        import tvm
        from tvm import te , auto_scheduler
        @auto_scheduler.register_workload
        def cv_fine_ltr_backy_workload(b, n, d, m):
            grad_output = te.placeholder((b, m+n , d), name='grad_output', dtype=dtype)
            t1 = te.placeholder((b, n+m, m*2), name='t1', dtype=dtype)
            k = te.reduce_axis((0, 2*m), name='k')
            output_shape = (b, n, d)
            algorithm_y_A = lambda l,i,j: te.sum(
                    te.if_then_else(k<m,
                        grad_output[l, m*(te.floordiv(i,m))+ k, j] * t1[l, m*(te.floordiv(i,m))+ k, te.floormod(i,m) +m],
                        grad_output[l, m*(te.floordiv(i,m))+ k, j] * t1[l, m*(te.floordiv(i,m))+ k, te.floormod(i,m)]
                    ),
                    axis = k,
                    where = (k >= te.floormod(i, m))
                )
            grad_t2 = te.compute(output_shape, algorithm_y_A, name='grad_t2_A')
            return [t1, grad_output, grad_t2]
        target = tvm.target.cuda(model=CUDA_MODEL, arch=CUDA_ARCH)
        tasky = auto_scheduler.SearchTask(
            func=cv_fine_ltr_backy_workload, args=(b, n, d, m), target=target
        )
        print("Computational DAG:")
        print(tasky.compute_dag)
        log_file = "aslogs/cv_fine_ltr_backy.json"
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
    def _get_lib_filename(dtype: str, device: str, functype: str, b,n,d,m):
        base_filename = 'funcs/cv_fine'
        return '{}_{}_{}_{}_{}_b{}_n{}_d{}_m{}.so'.format(
            base_filename, CUDA_MODEL, dtype, device, functype, b,n,d,m
        )

    @staticmethod
    def _save_compiled_function(f, dtype: str, device: str, functype: str, b,n,d,m):
        if not os.path.exists('funcs/'):
            os.makedirs('funcs/')
        f.export_library(DiagonaledMMCVFine._get_lib_filename(dtype, device, functype, b,n,d,m))

    @staticmethod
    def _load_compiled_function(dtype: str, device: str, functype: str, b,n,d,m):
        from tvm.runtime import load_module
        filename = DiagonaledMMCVFine._get_lib_filename(dtype, device, functype, b,n,d,m)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        potential_dirs = ['../../', '../', './', f'{current_dir}/', f'{current_dir}/../']
        for potential_dir in  potential_dirs:
            filepath = '{}{}'.format(potential_dir, filename)
            if os.path.isfile(filepath):
                print('Loading tvm binary from: {}'.format(filepath))
                return load_module(filepath)
        return None

    @staticmethod
    def _get_function(dtype: str, device: str, functype: str, b,n,d,m):
        args = (dtype, device, functype, b,n,d,m)
        if args not in DiagonaledMMCVFine.function_dict:
            f = DiagonaledMMCVFine._load_compiled_function(dtype, device, functype, b,n,d,m)
            if not f:
                print('Tvm binary not found. Compiling ...' + functype)
                if functype == 'forward':
                    f = DiagonaledMMCVFine._compile_function(dtype, device, b,n,d,m)
                elif functype == 'backx':
                    f = DiagonaledMMCVFine._compile_function_back_x(dtype, device, b,n,d,m)
                elif functype == 'backy':
                    f = DiagonaledMMCVFine._compile_function_back_y(dtype, device, b,n,d,m)
                DiagonaledMMCVFine._save_compiled_function(f, dtype, device, functype, b,n,d,m)
            from tvm.contrib import dlpack
            f_pytorch = dlpack.to_pytorch_func(f)
            DiagonaledMMCVFine.function_dict[args] = f_pytorch
        return DiagonaledMMCVFine.function_dict[args]

    @staticmethod
    def _diagonaled_mm(t1: torch.Tensor, t2: torch.Tensor, m: int):
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        b = t1.shape[0]
        n = t1.shape[1]
        d = t2.shape[2]
        Z = torch.zeros((b,n,d), dtype=t1.dtype, device=t1.device)
        t2_pad = torch.cat(
            (torch.zeros((b,m,d), dtype=t2.dtype, device=t2.device), t2),
            dim=1
        )
        _diagonaled_mm_function = DiagonaledMMCVFine._get_function(dtype, device, 'forward',b,n,d,m)
        _diagonaled_mm_function(t1, t2_pad, Z)
        return Z

    @staticmethod
    def _back_x(t2: torch.Tensor, grad_output: torch.Tensor, m: int):
        dtype = str(t2.dtype).split('.')[1]
        device = t2.device.type
        b = t2.shape[0]
        d = t2.shape[2]
        n = grad_output.shape[1]
        grad_t1 = t2.new_empty(b,n,2*m)
        t2_pad = torch.cat(
            (   torch.zeros((b,m,d), dtype=t2.dtype, device=t2.device),
                t2,),
            dim=1
        )
        _back_x_function = DiagonaledMMCVFine._get_function(dtype, device, 'backx',b,n,d,m)
        _back_x_function(t2_pad, grad_output, grad_t1)
        return grad_t1

    @staticmethod
    def _back_y(t1: torch.Tensor, grad_output: torch.Tensor, m: int):
        dtype = str(t1.dtype).split('.')[1]
        device = t1.device.type
        b = t1.shape[0]
        n = t1.shape[1]
        d = grad_output.shape[2]
        grad_t2_A = t1.new_empty(b,n,d)
        grad_output_pad = torch.cat(
            [   grad_output,
                torch.zeros((b,m,d), dtype=grad_output.dtype, device=grad_output.device)], 
            dim=1
        )
        t1_pad = torch.cat(
            [   t1,
                torch.zeros((b,m,m*2), dtype=t1.dtype, device=t1.device)],
            dim=1
        )
        _back_y_function = DiagonaledMMCVFine._get_function(dtype, device, 'backy',b,n,d,m)
        _back_y_function(t1_pad, grad_output_pad, grad_t2_A)
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
    def forward(ctx, t1: torch.Tensor, t2: torch.Tensor, m: int):
        ctx.save_for_backward(t1, t2)
        ctx.m = m
        t1 = DiagonaledMMCVFine._prepare_tensors(t1)
        t2 = DiagonaledMMCVFine._prepare_tensors(t2)
        output = DiagonaledMMCVFine._diagonaled_mm(t1, t2, m)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        t1, t2 = ctx.saved_tensors
        m = ctx.m
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_output = DiagonaledMMCVFine._prepare_tensors(grad_output)
        t1 = DiagonaledMMCVFine._prepare_tensors(t1)
        t2 = DiagonaledMMCVFine._prepare_tensors(t2)
        grad_t1 = DiagonaledMMCVFine._back_x(t2, grad_output, m)
        grad_t2 = DiagonaledMMCVFine._back_y(t1, grad_output, m)
        return grad_t1, grad_t2, None

diagonaled_mm_cv_fine_ltr = DiagonaledMMCVFine.apply