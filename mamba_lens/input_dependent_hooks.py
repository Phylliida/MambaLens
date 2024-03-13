# some of this code is modified from https://github.com/neelnanda-io/TransformerLens

from __future__ import annotations
from typing import Callable, List, Iterator, Tuple, Union
from contextlib import contextmanager

from transformer_lens.HookedTransformer import HookedRootModule
from transformer_lens.hook_points import HookPoint, NamesFilter

def hook_was_added_by_hookpoint(hook) -> bool:
    '''
    Internal function used for removing internal pytorch hooks directly,
    this tells us if the internal hook was added by a hook point
    This is important as pytorch uses some hook points in internal stuff
    and we don't want to remove those
    '''
    return "HookPoint." in str(repr(hook))

def clean_ghost_hooks_from_hook_point(dir : str, hook_point : HookPoint):
    '''
    This function will manually remove any "ghost" hooks from the pytorch module directly
    (these can occur and stick around if you interrupted code after the time the hook point was added
    but before the handle was added to the list)
    '''
    if dir in ['fwd', 'both']:
        for k, v in list(hook_point._forward_hooks.items()):
            if hook_was_added_by_hookpoint(v):
                print("leftover ghost hook", hook_point.name, k, v.__name__, "removing")
                del hook_point._forward_hooks[k]
            
        for k, v in list(hook_point._forward_pre_hooks.items()):
            if hook_was_added_by_hookpoint(v):
                print("leftover ghost hook", hook_point.name, k, v.__name__, "removing")
                del hook_point._forward_pre_hooks[k]
    elif dir in ['bwd', 'both']:
        for k, v in list(hook_point._backward_hooks.items()):
            if hook_was_added_by_hookpoint(v):
                print("leftover ghost hook", hook_point.name, k, v.__name__, "removing")
                del hook_point._backward_hooks[k]

def clean_hooks_from_hook_point(dir : str, hook_point : HookPoint):
    '''
    remove all hooks from the hook_point
    This function will manually remove any remaining hooks from the pytorch module directly
    (these can stick around if you interrupted code after the time the hook point was added
    but before the handle was added to the list)
    '''
    hook_point.remove_hooks(dir=dir, including_permanent=True, level=None)
    clean_ghost_hooks_from_hook_point(dir=dir, hook_point=hook_point)
    

def clean_hooks(model : HookedRootModule):
    '''
    remove all hooks from the model
    sometimes remove_all_hook_fns and reset_hooks won't suffice,
    because you interrupted the code between the time where the hook is added to python
    but before python has a chance to add a handle to the list HookPoint holds onto

    This function will manually remove any remaining hooks from the pytorch module directly
    '''
    model.reset_hooks(including_permanent=True, level=None)
    model.remove_all_hook_fns(including_permanent=True, level=None)
    # extra stuff in case that wasn't everything
    # this can happen if you interrupt between the time the hook is added to python
    # but before python has a chance to add the handle to the list HookPoint holds onto

    for name, module in model.named_modules():
        clean_hooks_from_hook_point(dir='fwd', hook_name=name, hook_point=module)
        clean_hooks_from_hook_point(dir='bwd', hook_name=name, hook_point=module)

class InputDependentHookPoint(HookPoint):
    '''
    This is a HookPoint that creates child hook points depending on the input size
    For example, the recurrent state of an RNN
    
    THIS CLASS DOES NOT WORK WITH HookedRootModule or HookedTransformer!!

    It must be used with InputDependentHookedRootModule (or something that inherits from it)
    '''
    def __init__(self, make_input_dependent_postfixes):
        """
        Args:
            make_input_dependent_postfixes: A function. When provided a parameter called
                "input", this should return all postfixes needed for that input
                For example, if this is being used as an RNN hidden state, and 
                input is of size [batch, 5] make_input_dependent_postfixes could return
                [".0", ".1", ".2", ".3", ".4"]
        """
        super().__init__()
        self.hooks = {}
        self.make_input_dependent_postfixes = make_input_dependent_postfixes

    def add_input_dependent_hooks(self, input):
        """
        Adds child hooks according to the provided input
        Args:
            input: The input to the model

        Note:
            Typically this will be called by your InputDependentHookedRootModule and you don't need to worry about this
        """
        for postfix in self.make_input_dependent_postfixes(input=input):
            if not postfix in self.hooks:
                postfix_hook = HookPoint()
                postfix_hook.name = self.name + postfix
                self.hooks[postfix] = postfix_hook
                yield self.hooks[postfix].name, self.hooks[postfix]
    
    def __call__(self, value, postfix):
        """
        Call this hook on the child corresponding to the provided postfix
        Args:
            input: The input to the model
            postfix: the postfix of which child we are calling on. This should correspond to
                the postfixes returned by make_input_dependent_postfixes
        
        Example usage:
        ```
        def make_postfix(l):
            return f".{l}"

        def make_input_dependent_postfixes(input):
            Batch, L = input.size()
            for l in range(L):
                postfix = make_postfix(l=l)
                yield postfix
        
        # In a class that inherits from InputDependentHookedRootModule:
        
            # In __init__, before calling self.setup()
                
                self.hook_h = InputDependentHookPoint(make_input_dependent_postfixes)

            # In forward:
                h = torch.zeros([batch,internal_dim], device=self.cfg.device)
                for l in range(seq_len):
                    # some internal rnn logic
                    h        =    update_hidden_state(h)
                    
                    # call the hook
                    postfix = make_postfix(l=l)
                    h        = self.hook_h(h, postfix=postfix)
        ```
        """
        if postfix in self.hooks:
            input_dependent_hook = self.hooks[postfix]
            return input_dependent_hook(value)
        else: # not setup, either forward was called by itself or we are not using this hook
            return value

    def remove_hooks(self, dir="fwd", including_permanent=False, level=None) -> None:
        '''
        Removes hooks on all of the children hooks of this input dependent hook
        '''
        removing_ghost_hooks = level is None and including_permanent
        for child_hook in self.hooks.values():
            child_hook.remove_hooks(dir=dir, including_permanent=including_permanent, level=level)
            if removing_ghost_hooks: # helps ensure nothing is left around
                clean_ghost_hooks_from_hook_point(dir=dir, hook_point=child_hook)
        # this shouldn't have hooks, but in case it does, remove them too
        super().remove_hooks(dir=dir, including_permanent=including_permanent, level=level)
        if removing_ghost_hooks:
            clean_ghost_hooks_from_hook_point(dir=dir, hook_point=self)

class InputDependentHookedRootModule(HookedRootModule):
    '''
    This is a variation of HookedRootModule that supports having InputDependentHookPoint hooks
    in addition to having regular HookPoint hooks

    InputDependentHookPoint hooks are used when the number of hooks need to vary based on the input size

    For example, the recurrent state of an RNN
    '''

    def setup(self):
        """
        Sets up model.

        This function must be called in the model's `__init__` method AFTER defining all layers. It
        adds a parameter to each module containing its name, and builds a dictionary mapping module
        names to the module instances. It also initializes a hook dictionary for modules of type
        "HookPoint".

        Note that input dependent hook points will not be in this dictionary, they need to be setup
        by using input_dependent_hooks_context (don't worry about this, they will be automatically 
        setup if you call run_with_hooks or run_with_cache)
        """
        # setup hooks
        super().setup()

        # remove input_dependent hooks from hook_dict and mod_dict (multiple hooks for each of these will be added below during each call)
        for name, hook in self.input_dependent_hooks():
            if name in self.mod_dict:
                del self.mod_dict[name]
            if name in self.hook_dict:
                del self.hook_dict[name]
            hook.name = name
        
        self.did_setup_input_dependent_hooks = False
    

    def get_all_hook_names(self, *model_args, **model_kwargs):
        """
        Returns a list containing all the hook names of this model
        Requires inputs to the model, as the number of hooks depend on the input
        
        Returns:
            hook_names (list[str]): The names of all hooks
        """
        # fwd_hooks=None expands all input_dependent hooks
        # we need this context to get the input dependent hooks
        hook_names = []
        with self.input_dependent_hooks_context(*model_args, fwd_hooks=None, bwd_hooks=None, **model_kwargs):
            for name, hp in self.hook_dict.items():
                hook_names.append(name)
        return hook_names

    def hook_points(self):
        '''
        Returns every hook point of this model

        Returns:
            hooks (list[HookPoint]): All hooks
        '''
        
        if self.did_setup_input_dependent_hooks:
            return super().hook_points()
        # we need to also include the input_dependent ones
        else:
            input_dependent_hook_children = []
            for input_dependent_hook_name, input_dependent_hook in self.input_dependent_hooks():
                input_dependent_hook_children += list(input_dependent_hook.hooks.values())
            return list(super().hook_points()) + input_dependent_hook_children
        
    def run_with_hooks (
            self,
            *model_args,
            fwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
            bwd_hooks: List[Tuple[Union[str, Callable], Callable]] = [],
            reset_hooks_end=True,
            clear_contexts=False,
            **model_kwargs,
        ):
            """
            Runs the model with specified forward and backward hooks.

            Args:
                *model_args: Positional arguments for the model.
                fwd_hooks (List[Tuple[Union[str, Callable], Callable]]): A list of (name, hook), where name is
                    either the name of a hook point or a boolean function on hook names, and hook is the
                    function to add to that hook point. Hooks with names that evaluate to True are added
                    respectively.
                bwd_hooks (List[Tuple[Union[str, Callable], Callable]]): Same as fwd_hooks, but for the
                    backward pass.
                reset_hooks_end (bool): If True, all hooks are removed at the end, including those added
                    during this run. Default is True.
                clear_contexts (bool): If True, clears hook contexts whenever hooks are reset. Default is
                    False.
                **model_kwargs: Keyword arguments for the model.

            Note:
                If you want to use backward hooks, set `reset_hooks_end` to False, so the backward hooks
                remain active. This function only runs a forward pass.
            """
            # just call super, we need to override this function to ensure input_dependent hooks are setup
            with self.input_dependent_hooks_context(*model_args, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, **model_kwargs):
                res = super().run_with_hooks(
                    *model_args,
                    fwd_hooks=fwd_hooks,
                    bwd_hooks=bwd_hooks,
                    reset_hooks_end=reset_hooks_end,
                    clear_contexts=clear_contexts,
                    **model_kwargs
                )
                return res
    
    def run_with_cache(
            self,
            *model_args,
            names_filter: NamesFilter = None,
            device=None,
            remove_batch_dim=False,
            incl_bwd=False,
            reset_hooks_end=True,
            clear_contexts=False,
            **model_kwargs,
        ):
            """
            Runs the model and returns the model output and a Cache object.

            Args:
                *model_args: Positional arguments for the model.
                names_filter (NamesFilter, optional): A filter for which activations to cache. Accepts None, str,
                    list of str, or a function that takes a string and returns a bool. Defaults to None, which
                    means cache everything.
                device (str or torch.Device, optional): The device to cache activations on. Defaults to the
                    model device. WARNING: Setting a different device than the one used by the model leads to
                    significant performance degradation.
                remove_batch_dim (bool, optional): If True, removes the batch dimension when caching. Only
                    makes sense with batch_size=1 inputs. Defaults to False.
                incl_bwd (bool, optional): If True, calls backward on the model output and caches gradients
                    as well. Assumes that the model outputs a scalar (e.g., return_type="loss"). Custom loss
                    functions are not supported. Defaults to False.
                reset_hooks_end (bool, optional): If True, removes all hooks added by this function at the
                    end of the run. Defaults to True.
                clear_contexts (bool, optional): If True, clears hook contexts whenever hooks are reset.
                    Defaults to False.
                **model_kwargs: Keyword arguments for the model.

            Returns:
                tuple: A tuple containing the model output and a Cache object.
            """
            model_kwargs = dict(list(model_kwargs.items()))
            fwd_hooks = None
            if 'fwd_hooks' in model_kwargs:
                fwd_hooks = model_kwargs['fwd_hooks']
                del model_kwargs['fwd_hooks']
            bwd_hooks = None
            if 'bwd_hooks' in model_kwargs:
                bwd_hooks = model_kwargs['bwd_hooks']
                del model_kwargs['bwd_hooks']
            # need to wrap run_with_cache to setup input_dependent hooks
            setup_all_input_hooks = False

            # turn names_filter into a fwd_hooks for setup input dependent hooks stuff
            if names_filter is None:
                setup_all_input_hooks = True
            else:
                name_fake_hooks = [(name, None) for name in names_filter]
                if fwd_hooks is None:
                    fwd_hooks = name_fake_hooks
                else:
                    fwd_hooks = fwd_hooks + name_fake_hooks
                    
            with self.input_dependent_hooks_context(*model_args, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, setup_all_input_hooks=setup_all_input_hooks, **model_kwargs):
                res = super().run_with_cache(
                    *model_args,
                    names_filter=names_filter,
                    device=device,
                    remove_batch_dim=remove_batch_dim,
                    incl_bwd=incl_bwd,
                    reset_hooks_end=reset_hooks_end,
                    clear_contexts=clear_contexts,
                    **model_kwargs)
                return res
        
    def input_dependent_hooks(self) -> Iterator[Tuple[str, InputDependentHookPoint]]:
        """
        Returns:
            (name, module) Iterator[str, InputDependentHookPoint]: An iterator of tuples of (name, module) where module is an InputDependentHookPoint
        """
        for name, module in self.named_modules():
            if name == "":
                continue
            if "InputDependentHookPoint" in str(type(module)):
                yield name, module
    
    @contextmanager
    def input_dependent_hooks_context(self, *model_args, fwd_hooks, bwd_hooks, setup_all_input_hooks=False, **model_kwargs):
        """
        Used to initialize any hooks that depend on the size of the input
        Usually you should just use run_with_cache or run_with_hooks which will do this for you

        Args:
            *model_args: Positional arguments for the model.
            fwd_hooks (List[Tuple[Union[str, Callable], Callable]]): A list of (name, hook), where name is
                either the name of a hook point or a boolean function on hook names, and hook is the
                function to add to that hook point. Hooks with names that evaluate to True are added
                respectively.
            bwd_hooks (List[Tuple[Union[str, Callable], Callable]]): Same as fwd_hooks, but for the
                backward pass.
            **model_kwargs: Keyword arguments for the model.

        Note:
            This should be used as a context manager. For example:
            ```
            with model.input_dependent_hooks(input, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                model.forward(input)
            ```
            These hooks are created and destroyed each time you enter the context, so do not depend on them staying around!
        """
        
        try:
            self.setup_input_dependent_hooks(*model_args, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, setup_all_input_hooks=setup_all_input_hooks, **model_kwargs)
            yield self
        finally:
            self.cleanup_input_dependent_hooks()

    def setup_input_dependent_hooks(self, *model_args, fwd_hooks, bwd_hooks, setup_all_input_hooks=False, **model_kwargs):
        """
        Initializes any hooks that depend on the size of the input
        You should not use this, instead, use input_dependent_hooks_context which will call cleanup_input_dependent_hooks when it is done
        Args:
            *model_args: Positional arguments for the model.
            fwd_hooks (List[Tuple[Union[str, Callable], Callable]]): A list of (name, hook), where name is
                either the name of a hook point or a boolean function on hook names, and hook is the
                function to add to that hook point. Hooks with names that evaluate to True are added
                respectively.
            bwd_hooks (List[Tuple[Union[str, Callable], Callable]]): Same as fwd_hooks, but for the
                backward pass.
            setup_all_input_hooks (bool): If True, all input dependent hooks are setup. Otherwise, only
                hooks provided in fwd_hooks or bwd_hooks are setup.
            **model_kwargs: Keyword arguments for the model.
        
        Note:
            This function will treat the first input to model_args as input. If 
        """
        # various ways input might be encoded in the model
        if 'input' in model_kwargs:
            input = model_kwargs['input']
        elif len(model_args) > 0:
            input = model_args[0]
        elif 'model_args' in model_kwargs and len(model_kwargs['model_args']) > 0:
            input = model_kwargs['model_args'][0]
        else:
            raise Exception(f"Could not find input in args {model_args} and kwargs {model_kwargs}")
        
        # make sure input is ids and not a str
        if type(input) is str:
            input = self.tokenizer(input, return_tensors='pt')['input_ids']
        input_dependent_lookup = {}
        for name, hook in self.input_dependent_hooks():
            input_dependent_lookup[name] = hook
        input_dependent = []

        if fwd_hooks is not None:
            for name, hook in fwd_hooks:
                if type(name) == str and not name in self.mod_dict:
                    input_dependent.append(name)
                else:
                    setup_all_input_hooks = True # we don't know what things make this eval to true, so expand them all
        if bwd_hooks is not None:
            for name, hook in bwd_hooks:
                if type(name) == str and not name in self.mod_dict:
                    input_dependent.append(name)
                else:
                    setup_all_input_hooks = True # we don't know what things make this eval to true, so expand them all
        
        # Lookup what input dependent hooks we need to setup
        hooks_to_expand = []
        if setup_all_input_hooks:
            hooks_to_expand = list(self.input_dependent_hooks())
        else:
            for name in input_dependent:
                # look for any prefix-matches, if we have them we need to expand them 
                for input_dependent_name, input_dependent_hook in self.input_dependent_hooks():
                    if name.startswith(input_dependent_name):
                        hooks_to_expand.append((input_dependent_name, input_dependent_hook))
        
        # Set them up
        for name, hook in hooks_to_expand:
            for added_hook_name, added_hook in hook.add_input_dependent_hooks(input=input):
                self.mod_dict[added_hook_name] = added_hook
                self.hook_dict[added_hook_name] = added_hook

        self.did_setup_input_dependent_hooks = True
    
    def cleanup_input_dependent_hooks(self):
        """
        Removes any hooks created by setup_input_dependent_hooks
        You should not use this, instead, use input_dependent_hooks_context which will call cleanup_input_dependent_hooks when it is done
        """
        for name, hook in self.input_dependent_hooks():
            for input_dependent_hook_postfix, input_dependent_hook in hook.hooks.items():
                input_dependent_hook_name = input_dependent_hook.name
                if input_dependent_hook_name in self.mod_dict:
                    del self.mod_dict[input_dependent_hook_name]
                if input_dependent_hook_name in self.hook_dict:
                    del self.hook_dict[input_dependent_hook_name]
            # this helps ensure any leftover hooks are removed
            hook.remove_hooks(dir='fwd', including_permanent=True, level=None)
            hook.remove_hooks(dir='bwd', including_permanent=True, level=None)
            hook.hooks = {}
        self.did_setup_input_dependent_hooks = False
