??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8??	
?
conv2d_54/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_54/kernel
}
$conv2d_54/kernel/Read/ReadVariableOpReadVariableOpconv2d_54/kernel*&
_output_shapes
: *
dtype0
t
conv2d_54/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_54/bias
m
"conv2d_54/bias/Read/ReadVariableOpReadVariableOpconv2d_54/bias*
_output_shapes
: *
dtype0
?
conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
: 0*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:0*
dtype0
|
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_51/kernel
u
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel* 
_output_shapes
:
??*
dtype0
r
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_51/bias
k
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/conv2d_54/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_54/kernel/m
?
+Adam/conv2d_54/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_54/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_54/bias/m
{
)Adam/conv2d_54/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*(
shared_nameAdam/conv2d_55/kernel/m
?
+Adam/conv2d_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/m*&
_output_shapes
: 0*
dtype0
?
Adam/conv2d_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_55/bias/m
{
)Adam/conv2d_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/m*
_output_shapes
:0*
dtype0
?
Adam/dense_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_51/kernel/m
?
*Adam/dense_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_51/bias/m
y
(Adam/dense_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_54/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_54/kernel/v
?
+Adam/conv2d_54/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_54/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_54/bias/v
{
)Adam/conv2d_54/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_54/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0*(
shared_nameAdam/conv2d_55/kernel/v
?
+Adam/conv2d_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/v*&
_output_shapes
: 0*
dtype0
?
Adam/conv2d_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*&
shared_nameAdam/conv2d_55/bias/v
{
)Adam/conv2d_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/v*
_output_shapes
:0*
dtype0
?
Adam/dense_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_51/kernel/v
?
*Adam/dense_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_51/bias/v
y
(Adam/dense_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_51/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?0
value?0B?0 B?0
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
R
#	variables
$trainable_variables
%regularization_losses
&	keras_api
R
'	variables
(trainable_variables
)regularization_losses
*	keras_api
R
+	variables
,trainable_variables
-regularization_losses
.	keras_api
h

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemrmsmtmu/mv0mwvxvyvzv{/v|0v}
*
0
1
2
3
/4
05
*
0
1
2
3
/4
05
 
?
:non_trainable_variables
;layer_regularization_losses

<layers

	variables
=layer_metrics
trainable_variables
regularization_losses
>metrics
 
\Z
VARIABLE_VALUEconv2d_54/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_54/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?non_trainable_variables
@layer_regularization_losses

Alayers
	variables
trainable_variables
Bmetrics
regularization_losses
Clayer_metrics
 
 
 
?
Dnon_trainable_variables
Elayer_regularization_losses

Flayers
	variables
trainable_variables
Gmetrics
regularization_losses
Hlayer_metrics
 
 
 
?
Inon_trainable_variables
Jlayer_regularization_losses

Klayers
	variables
trainable_variables
Lmetrics
regularization_losses
Mlayer_metrics
\Z
VARIABLE_VALUEconv2d_55/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_55/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Nnon_trainable_variables
Olayer_regularization_losses

Players
	variables
 trainable_variables
Qmetrics
!regularization_losses
Rlayer_metrics
 
 
 
?
Snon_trainable_variables
Tlayer_regularization_losses

Ulayers
#	variables
$trainable_variables
Vmetrics
%regularization_losses
Wlayer_metrics
 
 
 
?
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
'	variables
(trainable_variables
[metrics
)regularization_losses
\layer_metrics
 
 
 
?
]non_trainable_variables
^layer_regularization_losses

_layers
+	variables
,trainable_variables
`metrics
-regularization_losses
alayer_metrics
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

/0
01

/0
01
 
?
bnon_trainable_variables
clayer_regularization_losses

dlayers
1	variables
2trainable_variables
emetrics
3regularization_losses
flayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
0
1
2
3
4
5
6
7
 

g0
h1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	itotal
	jcount
k	variables
l	keras_api
D
	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

i0
j1

k	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

m0
n1

p	variables
}
VARIABLE_VALUEAdam/conv2d_54/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_54/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_55/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_55/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_51/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_51/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_54/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_54/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_55/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_55/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_51/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_51/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_54_inputPlaceholder*0
_output_shapes
:?????????(?*
dtype0*%
shape:?????????(?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_54_inputconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasdense_51/kerneldense_51/bias*
Tin
	2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*-
f(R&
$__inference_signature_wrapper_161582
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_54/kernel/Read/ReadVariableOp"conv2d_54/bias/Read/ReadVariableOp$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_54/kernel/m/Read/ReadVariableOp)Adam/conv2d_54/bias/m/Read/ReadVariableOp+Adam/conv2d_55/kernel/m/Read/ReadVariableOp)Adam/conv2d_55/bias/m/Read/ReadVariableOp*Adam/dense_51/kernel/m/Read/ReadVariableOp(Adam/dense_51/bias/m/Read/ReadVariableOp+Adam/conv2d_54/kernel/v/Read/ReadVariableOp)Adam/conv2d_54/bias/v/Read/ReadVariableOp+Adam/conv2d_55/kernel/v/Read/ReadVariableOp)Adam/conv2d_55/bias/v/Read/ReadVariableOp*Adam/dense_51/kernel/v/Read/ReadVariableOp(Adam/dense_51/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*(
f#R!
__inference__traced_save_162033
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_54/kernelconv2d_54/biasconv2d_55/kernelconv2d_55/biasdense_51/kerneldense_51/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_54/kernel/mAdam/conv2d_54/bias/mAdam/conv2d_55/kernel/mAdam/conv2d_55/bias/mAdam/dense_51/kernel/mAdam/dense_51/bias/mAdam/conv2d_54/kernel/vAdam/conv2d_54/bias/vAdam/conv2d_55/kernel/vAdam/conv2d_55/bias/vAdam/dense_51/kernel/vAdam/dense_51/bias/v*'
Tin 
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*+
f&R$
"__inference__traced_restore_162126??
?	
?
.__inference_sequential_87_layer_call_fn_161756

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_87_layer_call_and_return_conditional_losses_1615082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????(?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?L
?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161508

inputs
conv2d_54_161455
conv2d_54_161457
conv2d_55_161462
conv2d_55_161464
dense_51_161470
dense_51_161472
identity??!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_161455conv2d_54_161457*
Tin
2*
Tout
2*0
_output_shapes
:?????????'? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_1610982#
!conv2d_54/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_1611142"
 max_pooling2d_54/PartitionedCall?
dropout_54/PartitionedCallPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_54_layer_call_and_return_conditional_losses_1611972
dropout_54/PartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_54/PartitionedCall:output:0conv2d_55_161462conv2d_55_161464*
Tin
2*
Tout
2*0
_output_shapes
:??????????0*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_1611482#
!conv2d_55/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_1611642"
 max_pooling2d_55/PartitionedCall?
dropout_55/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_1612332
dropout_55/PartitionedCall?
flatten_27/PartitionedCallPartitionedCall#dropout_55/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_27_layer_call_and_return_conditional_losses_1612522
flatten_27/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_51_161470dense_51_161472*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_51_layer_call_and_return_conditional_losses_1612712"
 dense_51/StatefulPartitionedCall?
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161455*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/add?
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161457*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/add?
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161462*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/add?
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161464*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/add?
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall:X T
0
_output_shapes
:?????????(?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
h
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_161164

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
p
__inference_loss_fn_0_161886?
;conv2d_54_kernel_regularizer_square_readvariableop_resource
identity??
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_54_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/addg
IdentityIdentity$conv2d_54/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
b
F__inference_flatten_27_layer_call_and_return_conditional_losses_161848

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?
G
+__inference_flatten_27_layer_call_fn_161853

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_27_layer_call_and_return_conditional_losses_1612522
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_161114

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?L
?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161376
conv2d_54_input
conv2d_54_161323
conv2d_54_161325
conv2d_55_161330
conv2d_55_161332
dense_51_161338
dense_51_161340
identity??!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputconv2d_54_161323conv2d_54_161325*
Tin
2*
Tout
2*0
_output_shapes
:?????????'? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_1610982#
!conv2d_54/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_1611142"
 max_pooling2d_54/PartitionedCall?
dropout_54/PartitionedCallPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_54_layer_call_and_return_conditional_losses_1611972
dropout_54/PartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_54/PartitionedCall:output:0conv2d_55_161330conv2d_55_161332*
Tin
2*
Tout
2*0
_output_shapes
:??????????0*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_1611482#
!conv2d_55/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_1611642"
 max_pooling2d_55/PartitionedCall?
dropout_55/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_1612332
dropout_55/PartitionedCall?
flatten_27/PartitionedCallPartitionedCall#dropout_55/PartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_27_layer_call_and_return_conditional_losses_1612522
flatten_27/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_51_161338dense_51_161340*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_51_layer_call_and_return_conditional_losses_1612712"
 dense_51/StatefulPartitionedCall?
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161323*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/add?
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161325*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/add?
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161330*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/add?
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161332*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/add?
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall:a ]
0
_output_shapes
:?????????(?
)
_user_specified_nameconv2d_54_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
G
+__inference_dropout_54_layer_call_fn_161799

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_54_layer_call_and_return_conditional_losses_1611972
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
G
+__inference_dropout_55_layer_call_fn_161842

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_1612332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????	c02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?
d
F__inference_dropout_55_layer_call_and_return_conditional_losses_161233

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	c02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????	c02

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?
e
F__inference_dropout_54_layer_call_and_return_conditional_losses_161784

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????? 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
? 
?
E__inference_conv2d_54_layer_call_and_return_conditional_losses_161098

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/add?
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/add?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????:::i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_55_layer_call_and_return_conditional_losses_161228

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????	c02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????	c0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????	c02
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????	c02
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????	c02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????	c02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?b
?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161659

inputs,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource
identity??
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2DConv2Dinputs'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????'? *
paddingVALID*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????'? 2
conv2d_54/BiasAdd
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*0
_output_shapes
:?????????'? 2
conv2d_54/Relu?
max_pooling2d_54/MaxPoolMaxPoolconv2d_54/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_54/MaxPooly
dropout_54/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_54/dropout/Const?
dropout_54/dropout/MulMul!max_pooling2d_54/MaxPool:output:0!dropout_54/dropout/Const:output:0*
T0*0
_output_shapes
:?????????? 2
dropout_54/dropout/Mul?
dropout_54/dropout/ShapeShape!max_pooling2d_54/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_54/dropout/Shape?
/dropout_54/dropout/random_uniform/RandomUniformRandomUniform!dropout_54/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????? *
dtype021
/dropout_54/dropout/random_uniform/RandomUniform?
!dropout_54/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_54/dropout/GreaterEqual/y?
dropout_54/dropout/GreaterEqualGreaterEqual8dropout_54/dropout/random_uniform/RandomUniform:output:0*dropout_54/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????? 2!
dropout_54/dropout/GreaterEqual?
dropout_54/dropout/CastCast#dropout_54/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????? 2
dropout_54/dropout/Cast?
dropout_54/dropout/Mul_1Muldropout_54/dropout/Mul:z:0dropout_54/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????? 2
dropout_54/dropout/Mul_1?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_55/Conv2D/ReadVariableOp?
conv2d_55/Conv2DConv2Ddropout_54/dropout/Mul_1:z:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingVALID*
strides
2
conv2d_55/Conv2D?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
conv2d_55/BiasAdd
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
conv2d_55/Relu?
max_pooling2d_55/MaxPoolMaxPoolconv2d_55/Relu:activations:0*/
_output_shapes
:?????????	c0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPooly
dropout_55/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_55/dropout/Const?
dropout_55/dropout/MulMul!max_pooling2d_55/MaxPool:output:0!dropout_55/dropout/Const:output:0*
T0*/
_output_shapes
:?????????	c02
dropout_55/dropout/Mul?
dropout_55/dropout/ShapeShape!max_pooling2d_55/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_55/dropout/Shape?
/dropout_55/dropout/random_uniform/RandomUniformRandomUniform!dropout_55/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????	c0*
dtype021
/dropout_55/dropout/random_uniform/RandomUniform?
!dropout_55/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!dropout_55/dropout/GreaterEqual/y?
dropout_55/dropout/GreaterEqualGreaterEqual8dropout_55/dropout/random_uniform/RandomUniform:output:0*dropout_55/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????	c02!
dropout_55/dropout/GreaterEqual?
dropout_55/dropout/CastCast#dropout_55/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????	c02
dropout_55/dropout/Cast?
dropout_55/dropout/Mul_1Muldropout_55/dropout/Mul:z:0dropout_55/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????	c02
dropout_55/dropout/Mul_1u
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_27/Const?
flatten_27/ReshapeReshapedropout_55/dropout/Mul_1:z:0flatten_27/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_27/Reshape?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMulflatten_27/Reshape:output:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_51/BiasAdd|
dense_51/SoftmaxSoftmaxdense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_51/Softmax?
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/add?
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/add?
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/add?
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/addn
IdentityIdentitydense_51/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?:::::::X T
0
_output_shapes
:?????????(?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?N
?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161722

inputs,
(conv2d_54_conv2d_readvariableop_resource-
)conv2d_54_biasadd_readvariableop_resource,
(conv2d_55_conv2d_readvariableop_resource-
)conv2d_55_biasadd_readvariableop_resource+
'dense_51_matmul_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource
identity??
conv2d_54/Conv2D/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_54/Conv2D/ReadVariableOp?
conv2d_54/Conv2DConv2Dinputs'conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????'? *
paddingVALID*
strides
2
conv2d_54/Conv2D?
 conv2d_54/BiasAdd/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_54/BiasAdd/ReadVariableOp?
conv2d_54/BiasAddBiasAddconv2d_54/Conv2D:output:0(conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????'? 2
conv2d_54/BiasAdd
conv2d_54/ReluReluconv2d_54/BiasAdd:output:0*
T0*0
_output_shapes
:?????????'? 2
conv2d_54/Relu?
max_pooling2d_54/MaxPoolMaxPoolconv2d_54/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_54/MaxPool?
dropout_54/IdentityIdentity!max_pooling2d_54/MaxPool:output:0*
T0*0
_output_shapes
:?????????? 2
dropout_54/Identity?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_55/Conv2D/ReadVariableOp?
conv2d_55/Conv2DConv2Ddropout_54/Identity:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingVALID*
strides
2
conv2d_55/Conv2D?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02"
 conv2d_55/BiasAdd/ReadVariableOp?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02
conv2d_55/BiasAdd
conv2d_55/ReluReluconv2d_55/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
conv2d_55/Relu?
max_pooling2d_55/MaxPoolMaxPoolconv2d_55/Relu:activations:0*/
_output_shapes
:?????????	c0*
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPool?
dropout_55/IdentityIdentity!max_pooling2d_55/MaxPool:output:0*
T0*/
_output_shapes
:?????????	c02
dropout_55/Identityu
flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_27/Const?
flatten_27/ReshapeReshapedropout_55/Identity:output:0flatten_27/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten_27/Reshape?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMulflatten_27/Reshape:output:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_51/BiasAdd|
dense_51/SoftmaxSoftmaxdense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_51/Softmax?
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/add?
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/add?
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/add?
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/addn
IdentityIdentitydense_51/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?:::::::X T
0
_output_shapes
:?????????(?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_dense_51_layer_call_and_return_conditional_losses_161271

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:::Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
M
1__inference_max_pooling2d_54_layer_call_fn_161120

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_1611142
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_54_layer_call_and_return_conditional_losses_161197

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????? 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????? 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?*
?
!__inference__wrapped_model_161070
conv2d_54_input:
6sequential_87_conv2d_54_conv2d_readvariableop_resource;
7sequential_87_conv2d_54_biasadd_readvariableop_resource:
6sequential_87_conv2d_55_conv2d_readvariableop_resource;
7sequential_87_conv2d_55_biasadd_readvariableop_resource9
5sequential_87_dense_51_matmul_readvariableop_resource:
6sequential_87_dense_51_biasadd_readvariableop_resource
identity??
-sequential_87/conv2d_54/Conv2D/ReadVariableOpReadVariableOp6sequential_87_conv2d_54_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_87/conv2d_54/Conv2D/ReadVariableOp?
sequential_87/conv2d_54/Conv2DConv2Dconv2d_54_input5sequential_87/conv2d_54/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????'? *
paddingVALID*
strides
2 
sequential_87/conv2d_54/Conv2D?
.sequential_87/conv2d_54/BiasAdd/ReadVariableOpReadVariableOp7sequential_87_conv2d_54_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_87/conv2d_54/BiasAdd/ReadVariableOp?
sequential_87/conv2d_54/BiasAddBiasAdd'sequential_87/conv2d_54/Conv2D:output:06sequential_87/conv2d_54/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????'? 2!
sequential_87/conv2d_54/BiasAdd?
sequential_87/conv2d_54/ReluRelu(sequential_87/conv2d_54/BiasAdd:output:0*
T0*0
_output_shapes
:?????????'? 2
sequential_87/conv2d_54/Relu?
&sequential_87/max_pooling2d_54/MaxPoolMaxPool*sequential_87/conv2d_54/Relu:activations:0*0
_output_shapes
:?????????? *
ksize
*
paddingVALID*
strides
2(
&sequential_87/max_pooling2d_54/MaxPool?
!sequential_87/dropout_54/IdentityIdentity/sequential_87/max_pooling2d_54/MaxPool:output:0*
T0*0
_output_shapes
:?????????? 2#
!sequential_87/dropout_54/Identity?
-sequential_87/conv2d_55/Conv2D/ReadVariableOpReadVariableOp6sequential_87_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02/
-sequential_87/conv2d_55/Conv2D/ReadVariableOp?
sequential_87/conv2d_55/Conv2DConv2D*sequential_87/dropout_54/Identity:output:05sequential_87/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????0*
paddingVALID*
strides
2 
sequential_87/conv2d_55/Conv2D?
.sequential_87/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp7sequential_87_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype020
.sequential_87/conv2d_55/BiasAdd/ReadVariableOp?
sequential_87/conv2d_55/BiasAddBiasAdd'sequential_87/conv2d_55/Conv2D:output:06sequential_87/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????02!
sequential_87/conv2d_55/BiasAdd?
sequential_87/conv2d_55/ReluRelu(sequential_87/conv2d_55/BiasAdd:output:0*
T0*0
_output_shapes
:??????????02
sequential_87/conv2d_55/Relu?
&sequential_87/max_pooling2d_55/MaxPoolMaxPool*sequential_87/conv2d_55/Relu:activations:0*/
_output_shapes
:?????????	c0*
ksize
*
paddingVALID*
strides
2(
&sequential_87/max_pooling2d_55/MaxPool?
!sequential_87/dropout_55/IdentityIdentity/sequential_87/max_pooling2d_55/MaxPool:output:0*
T0*/
_output_shapes
:?????????	c02#
!sequential_87/dropout_55/Identity?
sequential_87/flatten_27/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2 
sequential_87/flatten_27/Const?
 sequential_87/flatten_27/ReshapeReshape*sequential_87/dropout_55/Identity:output:0'sequential_87/flatten_27/Const:output:0*
T0*)
_output_shapes
:???????????2"
 sequential_87/flatten_27/Reshape?
,sequential_87/dense_51/MatMul/ReadVariableOpReadVariableOp5sequential_87_dense_51_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,sequential_87/dense_51/MatMul/ReadVariableOp?
sequential_87/dense_51/MatMulMatMul)sequential_87/flatten_27/Reshape:output:04sequential_87/dense_51/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_87/dense_51/MatMul?
-sequential_87/dense_51/BiasAdd/ReadVariableOpReadVariableOp6sequential_87_dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_87/dense_51/BiasAdd/ReadVariableOp?
sequential_87/dense_51/BiasAddBiasAdd'sequential_87/dense_51/MatMul:product:05sequential_87/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_87/dense_51/BiasAdd?
sequential_87/dense_51/SoftmaxSoftmax'sequential_87/dense_51/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential_87/dense_51/Softmax|
IdentityIdentity(sequential_87/dense_51/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?:::::::a ]
0
_output_shapes
:?????????(?
)
_user_specified_nameconv2d_54_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
.__inference_sequential_87_layer_call_fn_161739

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_87_layer_call_and_return_conditional_losses_1614352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????(?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
M
1__inference_max_pooling2d_55_layer_call_fn_161170

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_1611642
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_dense_51_layer_call_fn_161873

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_51_layer_call_and_return_conditional_losses_1612712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
d
+__inference_dropout_55_layer_call_fn_161837

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_1612282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????	c02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	c022
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
? 
?
E__inference_conv2d_55_layer_call_and_return_conditional_losses_161148

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????02	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02
Relu?
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/add?
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/add?
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? :::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?O
?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161435

inputs
conv2d_54_161382
conv2d_54_161384
conv2d_55_161389
conv2d_55_161391
dense_51_161397
dense_51_161399
identity??!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?"dropout_54/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_54_161382conv2d_54_161384*
Tin
2*
Tout
2*0
_output_shapes
:?????????'? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_1610982#
!conv2d_54/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_1611142"
 max_pooling2d_54/PartitionedCall?
"dropout_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_54_layer_call_and_return_conditional_losses_1611922$
"dropout_54/StatefulPartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_54/StatefulPartitionedCall:output:0conv2d_55_161389conv2d_55_161391*
Tin
2*
Tout
2*0
_output_shapes
:??????????0*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_1611482#
!conv2d_55/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_1611642"
 max_pooling2d_55/PartitionedCall?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0#^dropout_54/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_1612282$
"dropout_55/StatefulPartitionedCall?
flatten_27/PartitionedCallPartitionedCall+dropout_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_27_layer_call_and_return_conditional_losses_1612522
flatten_27/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_51_161397dense_51_161399*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_51_layer_call_and_return_conditional_losses_1612712"
 dense_51/StatefulPartitionedCall?
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161382*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/add?
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161384*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/add?
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161389*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/add?
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161391*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/add?
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall#^dropout_54/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2H
"dropout_54/StatefulPartitionedCall"dropout_54/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall:X T
0
_output_shapes
:?????????(?
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
F__inference_flatten_27_layer_call_and_return_conditional_losses_161252

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?E
?
__inference__traced_save_162033
file_prefix/
+savev2_conv2d_54_kernel_read_readvariableop-
)savev2_conv2d_54_bias_read_readvariableop/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_54_kernel_m_read_readvariableop4
0savev2_adam_conv2d_54_bias_m_read_readvariableop6
2savev2_adam_conv2d_55_kernel_m_read_readvariableop4
0savev2_adam_conv2d_55_bias_m_read_readvariableop5
1savev2_adam_dense_51_kernel_m_read_readvariableop3
/savev2_adam_dense_51_bias_m_read_readvariableop6
2savev2_adam_conv2d_54_kernel_v_read_readvariableop4
0savev2_adam_conv2d_54_bias_v_read_readvariableop6
2savev2_adam_conv2d_55_kernel_v_read_readvariableop4
0savev2_adam_conv2d_55_bias_v_read_readvariableop5
1savev2_adam_dense_51_kernel_v_read_readvariableop3
/savev2_adam_dense_51_bias_v_read_readvariableop
savev2_1_const

identity_1??MergeV2Checkpoints?SaveV2?SaveV2_1?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_99043dabf8804e45bf140ca0970aa982/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_54_kernel_read_readvariableop)savev2_conv2d_54_bias_read_readvariableop+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_54_kernel_m_read_readvariableop0savev2_adam_conv2d_54_bias_m_read_readvariableop2savev2_adam_conv2d_55_kernel_m_read_readvariableop0savev2_adam_conv2d_55_bias_m_read_readvariableop1savev2_adam_dense_51_kernel_m_read_readvariableop/savev2_adam_dense_51_bias_m_read_readvariableop2savev2_adam_conv2d_54_kernel_v_read_readvariableop0savev2_adam_conv2d_54_bias_v_read_readvariableop2savev2_adam_conv2d_55_kernel_v_read_readvariableop0savev2_adam_conv2d_55_bias_v_read_readvariableop1savev2_adam_dense_51_kernel_v_read_readvariableop/savev2_adam_dense_51_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *)
dtypes
2	2
SaveV2?
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard?
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1?
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names?
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices?
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity?

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : 0:0:
??:: : : : : : : : : : : : 0:0:
??:: : : 0:0:
??:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:&"
 
_output_shapes
:
??: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:&"
 
_output_shapes
:
??: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: 0: 

_output_shapes
:0:&"
 
_output_shapes
:
??: 

_output_shapes
::

_output_shapes
: 
?

*__inference_conv2d_55_layer_call_fn_161158

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_1611482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_55_layer_call_and_return_conditional_losses_161827

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????	c02
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????	c0*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????	c02
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????	c02
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????	c02
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????	c02

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?	
?
.__inference_sequential_87_layer_call_fn_161450
conv2d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_87_layer_call_and_return_conditional_losses_1614352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:?????????(?
)
_user_specified_nameconv2d_54_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?O
?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161320
conv2d_54_input
conv2d_54_161174
conv2d_54_161176
conv2d_55_161210
conv2d_55_161212
dense_51_161282
dense_51_161284
identity??!conv2d_54/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?"dropout_54/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?
!conv2d_54/StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputconv2d_54_161174conv2d_54_161176*
Tin
2*
Tout
2*0
_output_shapes
:?????????'? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_1610982#
!conv2d_54/StatefulPartitionedCall?
 max_pooling2d_54/PartitionedCallPartitionedCall*conv2d_54/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_1611142"
 max_pooling2d_54/PartitionedCall?
"dropout_54/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_54/PartitionedCall:output:0*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_54_layer_call_and_return_conditional_losses_1611922$
"dropout_54/StatefulPartitionedCall?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_54/StatefulPartitionedCall:output:0conv2d_55_161210conv2d_55_161212*
Tin
2*
Tout
2*0
_output_shapes
:??????????0*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_55_layer_call_and_return_conditional_losses_1611482#
!conv2d_55/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*U
fPRN
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_1611642"
 max_pooling2d_55/PartitionedCall?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0#^dropout_54/StatefulPartitionedCall*
Tin
2*
Tout
2*/
_output_shapes
:?????????	c0* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_55_layer_call_and_return_conditional_losses_1612282$
"dropout_55/StatefulPartitionedCall?
flatten_27/PartitionedCallPartitionedCall+dropout_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*)
_output_shapes
:???????????* 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_flatten_27_layer_call_and_return_conditional_losses_1612522
flatten_27/PartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall#flatten_27/PartitionedCall:output:0dense_51_161282dense_51_161284*
Tin
2*
Tout
2*'
_output_shapes
:?????????*$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_dense_51_layer_call_and_return_conditional_losses_1612712"
 dense_51/StatefulPartitionedCall?
2conv2d_54/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161174*&
_output_shapes
: *
dtype024
2conv2d_54/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_54/kernel/Regularizer/SquareSquare:conv2d_54/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2%
#conv2d_54/kernel/Regularizer/Square?
"conv2d_54/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_54/kernel/Regularizer/Const?
 conv2d_54/kernel/Regularizer/SumSum'conv2d_54/kernel/Regularizer/Square:y:0+conv2d_54/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/Sum?
"conv2d_54/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_54/kernel/Regularizer/mul/x?
 conv2d_54/kernel/Regularizer/mulMul+conv2d_54/kernel/Regularizer/mul/x:output:0)conv2d_54/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/mul?
"conv2d_54/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_54/kernel/Regularizer/add/x?
 conv2d_54/kernel/Regularizer/addAddV2+conv2d_54/kernel/Regularizer/add/x:output:0$conv2d_54/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_54/kernel/Regularizer/add?
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_54_161176*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/add?
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161210*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/add?
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_55_161212*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/add?
IdentityIdentity)dense_51/StatefulPartitionedCall:output:0"^conv2d_54/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall#^dropout_54/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::2F
!conv2d_54/StatefulPartitionedCall!conv2d_54/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2H
"dropout_54/StatefulPartitionedCall"dropout_54/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall:a ]
0
_output_shapes
:?????????(?
)
_user_specified_nameconv2d_54_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
D__inference_dense_51_layer_call_and_return_conditional_losses_161864

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*0
_input_shapes
:???????????:::Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?	
?
.__inference_sequential_87_layer_call_fn_161523
conv2d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8*R
fMRK
I__inference_sequential_87_layer_call_and_return_conditional_losses_1615082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:?????????(?
)
_user_specified_nameconv2d_54_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
e
F__inference_dropout_54_layer_call_and_return_conditional_losses_161192

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????? 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
n
__inference_loss_fn_1_161899=
9conv2d_54_bias_regularizer_square_readvariableop_resource
identity??
0conv2d_54/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_54_bias_regularizer_square_readvariableop_resource*
_output_shapes
: *
dtype022
0conv2d_54/bias/Regularizer/Square/ReadVariableOp?
!conv2d_54/bias/Regularizer/SquareSquare8conv2d_54/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
: 2#
!conv2d_54/bias/Regularizer/Square?
 conv2d_54/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_54/bias/Regularizer/Const?
conv2d_54/bias/Regularizer/SumSum%conv2d_54/bias/Regularizer/Square:y:0)conv2d_54/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/Sum?
 conv2d_54/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_54/bias/Regularizer/mul/x?
conv2d_54/bias/Regularizer/mulMul)conv2d_54/bias/Regularizer/mul/x:output:0'conv2d_54/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/mul?
 conv2d_54/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_54/bias/Regularizer/add/x?
conv2d_54/bias/Regularizer/addAddV2)conv2d_54/bias/Regularizer/add/x:output:0"conv2d_54/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_54/bias/Regularizer/adde
IdentityIdentity"conv2d_54/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
n
__inference_loss_fn_3_161925=
9conv2d_55_bias_regularizer_square_readvariableop_resource
identity??
0conv2d_55/bias/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_55_bias_regularizer_square_readvariableop_resource*
_output_shapes
:0*
dtype022
0conv2d_55/bias/Regularizer/Square/ReadVariableOp?
!conv2d_55/bias/Regularizer/SquareSquare8conv2d_55/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:02#
!conv2d_55/bias/Regularizer/Square?
 conv2d_55/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 conv2d_55/bias/Regularizer/Const?
conv2d_55/bias/Regularizer/SumSum%conv2d_55/bias/Regularizer/Square:y:0)conv2d_55/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/Sum?
 conv2d_55/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2"
 conv2d_55/bias/Regularizer/mul/x?
conv2d_55/bias/Regularizer/mulMul)conv2d_55/bias/Regularizer/mul/x:output:0'conv2d_55/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/mul?
 conv2d_55/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 conv2d_55/bias/Regularizer/add/x?
conv2d_55/bias/Regularizer/addAddV2)conv2d_55/bias/Regularizer/add/x:output:0"conv2d_55/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
conv2d_55/bias/Regularizer/adde
IdentityIdentity"conv2d_55/bias/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
d
F__inference_dropout_55_layer_call_and_return_conditional_losses_161832

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????	c02

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????	c02

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????	c0:W S
/
_output_shapes
:?????????	c0
 
_user_specified_nameinputs
?

*__inference_conv2d_54_layer_call_fn_161108

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
**
config_proto

CPU

GPU 2J 8*N
fIRG
E__inference_conv2d_54_layer_call_and_return_conditional_losses_1610982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
?
d
F__inference_dropout_54_layer_call_and_return_conditional_losses_161789

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????? 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????? 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?w
?
"__inference__traced_restore_162126
file_prefix%
!assignvariableop_conv2d_54_kernel%
!assignvariableop_1_conv2d_54_bias'
#assignvariableop_2_conv2d_55_kernel%
!assignvariableop_3_conv2d_55_bias&
"assignvariableop_4_dense_51_kernel$
 assignvariableop_5_dense_51_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count
assignvariableop_13_total_1
assignvariableop_14_count_1/
+assignvariableop_15_adam_conv2d_54_kernel_m-
)assignvariableop_16_adam_conv2d_54_bias_m/
+assignvariableop_17_adam_conv2d_55_kernel_m-
)assignvariableop_18_adam_conv2d_55_bias_m.
*assignvariableop_19_adam_dense_51_kernel_m,
(assignvariableop_20_adam_dense_51_bias_m/
+assignvariableop_21_adam_conv2d_54_kernel_v-
)assignvariableop_22_adam_conv2d_54_bias_v/
+assignvariableop_23_adam_conv2d_55_kernel_v-
)assignvariableop_24_adam_conv2d_55_bias_v.
*assignvariableop_25_adam_dense_51_kernel_v,
(assignvariableop_26_adam_dense_51_bias_v
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	RestoreV2?RestoreV2_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_54_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_54_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_55_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_55_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_51_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_51_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp+assignvariableop_15_adam_conv2d_54_kernel_mIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_conv2d_54_bias_mIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_55_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_55_bias_mIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_51_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_51_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_54_kernel_vIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_54_bias_vIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_55_kernel_vIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_55_bias_vIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_51_kernel_vIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_51_bias_vIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26?
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names?
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices?
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27?
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*?
_input_shapesp
n: :::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
p
__inference_loss_fn_2_161912?
;conv2d_55_kernel_regularizer_square_readvariableop_resource
identity??
2conv2d_55/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_55_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_55/kernel/Regularizer/Square/ReadVariableOp?
#conv2d_55/kernel/Regularizer/SquareSquare:conv2d_55/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02%
#conv2d_55/kernel/Regularizer/Square?
"conv2d_55/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"conv2d_55/kernel/Regularizer/Const?
 conv2d_55/kernel/Regularizer/SumSum'conv2d_55/kernel/Regularizer/Square:y:0+conv2d_55/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/Sum?
"conv2d_55/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2$
"conv2d_55/kernel/Regularizer/mul/x?
 conv2d_55/kernel/Regularizer/mulMul+conv2d_55/kernel/Regularizer/mul/x:output:0)conv2d_55/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/mul?
"conv2d_55/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"conv2d_55/kernel/Regularizer/add/x?
 conv2d_55/kernel/Regularizer/addAddV2+conv2d_55/kernel/Regularizer/add/x:output:0$conv2d_55/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2"
 conv2d_55/kernel/Regularizer/addg
IdentityIdentity$conv2d_55/kernel/Regularizer/add:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:: 

_output_shapes
: 
?
?
$__inference_signature_wrapper_161582
conv2d_54_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_54_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*'
_output_shapes
:?????????*(
_read_only_resource_inputs

**
config_proto

CPU

GPU 2J 8**
f%R#
!__inference__wrapped_model_1610702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:?????????(?::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:?????????(?
)
_user_specified_nameconv2d_54_input:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
d
+__inference_dropout_54_layer_call_fn_161794

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 **
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_dropout_54_layer_call_and_return_conditional_losses_1611922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
T
conv2d_54_inputA
!serving_default_conv2d_54_input:0?????????(?<
dense_510
StatefulPartitionedCall:0?????????tensorflow/serving/predict:҉
?<
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
~_default_save_signature
__call__
+?&call_and_return_all_conditional_losses"?9
_tf_keras_sequential?9{"class_name": "Sequential", "name": "sequential_87", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_87", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 400, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_54", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 400, 1]}}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 400, 1]}, "is_graph_network": true, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_87", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 400, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_54", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 400, 1]}}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"class_name": "Conv2D", "name": "conv2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 400, 1]}, "stateful": false, "config": {"name": "conv2d_54", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 400, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 400, 1]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_54", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_54", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_54", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_54", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?


kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "conv2d_55", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 0.0010000000474974513}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 19, 199, 32]}}
?
#	variables
$trainable_variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_55", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten_27", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 42768}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 42768]}}
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemrmsmtmu/mv0mwvxvyvzv{/v|0v}"
	optimizer
J
0
1
2
3
/4
05"
trackable_list_wrapper
J
0
1
2
3
/4
05"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
?
:non_trainable_variables
;layer_regularization_losses

<layers

	variables
=layer_metrics
trainable_variables
regularization_losses
>metrics
__call__
~_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_54/kernel
: 2conv2d_54/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
?non_trainable_variables
@layer_regularization_losses

Alayers
	variables
trainable_variables
Bmetrics
regularization_losses
Clayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dnon_trainable_variables
Elayer_regularization_losses

Flayers
	variables
trainable_variables
Gmetrics
regularization_losses
Hlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Inon_trainable_variables
Jlayer_regularization_losses

Klayers
	variables
trainable_variables
Lmetrics
regularization_losses
Mlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( 02conv2d_55/kernel
:02conv2d_55/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
?
Nnon_trainable_variables
Olayer_regularization_losses

Players
	variables
 trainable_variables
Qmetrics
!regularization_losses
Rlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Snon_trainable_variables
Tlayer_regularization_losses

Ulayers
#	variables
$trainable_variables
Vmetrics
%regularization_losses
Wlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
'	variables
(trainable_variables
[metrics
)regularization_losses
\layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]non_trainable_variables
^layer_regularization_losses

_layers
+	variables
,trainable_variables
`metrics
-regularization_losses
alayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!
??2dense_51/kernel
:2dense_51/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
?
bnon_trainable_variables
clayer_regularization_losses

dlayers
1	variables
2trainable_variables
emetrics
3regularization_losses
flayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	itotal
	jcount
k	variables
l	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	mtotal
	ncount
o
_fn_kwargs
p	variables
q	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
k	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
/:- 2Adam/conv2d_54/kernel/m
!: 2Adam/conv2d_54/bias/m
/:- 02Adam/conv2d_55/kernel/m
!:02Adam/conv2d_55/bias/m
(:&
??2Adam/dense_51/kernel/m
 :2Adam/dense_51/bias/m
/:- 2Adam/conv2d_54/kernel/v
!: 2Adam/conv2d_54/bias/v
/:- 02Adam/conv2d_55/kernel/v
!:02Adam/conv2d_55/bias/v
(:&
??2Adam/dense_51/kernel/v
 :2Adam/dense_51/bias/v
?2?
!__inference__wrapped_model_161070?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/
conv2d_54_input?????????(?
?2?
.__inference_sequential_87_layer_call_fn_161523
.__inference_sequential_87_layer_call_fn_161739
.__inference_sequential_87_layer_call_fn_161450
.__inference_sequential_87_layer_call_fn_161756?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161659
I__inference_sequential_87_layer_call_and_return_conditional_losses_161376
I__inference_sequential_87_layer_call_and_return_conditional_losses_161722
I__inference_sequential_87_layer_call_and_return_conditional_losses_161320?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_54_layer_call_fn_161108?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
E__inference_conv2d_54_layer_call_and_return_conditional_losses_161098?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
1__inference_max_pooling2d_54_layer_call_fn_161120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_161114?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_dropout_54_layer_call_fn_161794
+__inference_dropout_54_layer_call_fn_161799?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_54_layer_call_and_return_conditional_losses_161784
F__inference_dropout_54_layer_call_and_return_conditional_losses_161789?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_conv2d_55_layer_call_fn_161158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
E__inference_conv2d_55_layer_call_and_return_conditional_losses_161148?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
1__inference_max_pooling2d_55_layer_call_fn_161170?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_161164?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_dropout_55_layer_call_fn_161842
+__inference_dropout_55_layer_call_fn_161837?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_dropout_55_layer_call_and_return_conditional_losses_161827
F__inference_dropout_55_layer_call_and_return_conditional_losses_161832?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_flatten_27_layer_call_fn_161853?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_flatten_27_layer_call_and_return_conditional_losses_161848?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_51_layer_call_fn_161873?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_51_layer_call_and_return_conditional_losses_161864?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_161886?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_161899?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_161912?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_161925?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
;B9
$__inference_signature_wrapper_161582conv2d_54_input?
!__inference__wrapped_model_161070?/0A?>
7?4
2?/
conv2d_54_input?????????(?
? "3?0
.
dense_51"?
dense_51??????????
E__inference_conv2d_54_layer_call_and_return_conditional_losses_161098?I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
*__inference_conv2d_54_layer_call_fn_161108?I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
E__inference_conv2d_55_layer_call_and_return_conditional_losses_161148?I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????0
? ?
*__inference_conv2d_55_layer_call_fn_161158?I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????0?
D__inference_dense_51_layer_call_and_return_conditional_losses_161864^/01?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? ~
)__inference_dense_51_layer_call_fn_161873Q/01?.
'?$
"?
inputs???????????
? "???????????
F__inference_dropout_54_layer_call_and_return_conditional_losses_161784n<?9
2?/
)?&
inputs?????????? 
p
? ".?+
$?!
0?????????? 
? ?
F__inference_dropout_54_layer_call_and_return_conditional_losses_161789n<?9
2?/
)?&
inputs?????????? 
p 
? ".?+
$?!
0?????????? 
? ?
+__inference_dropout_54_layer_call_fn_161794a<?9
2?/
)?&
inputs?????????? 
p
? "!??????????? ?
+__inference_dropout_54_layer_call_fn_161799a<?9
2?/
)?&
inputs?????????? 
p 
? "!??????????? ?
F__inference_dropout_55_layer_call_and_return_conditional_losses_161827l;?8
1?.
(?%
inputs?????????	c0
p
? "-?*
#? 
0?????????	c0
? ?
F__inference_dropout_55_layer_call_and_return_conditional_losses_161832l;?8
1?.
(?%
inputs?????????	c0
p 
? "-?*
#? 
0?????????	c0
? ?
+__inference_dropout_55_layer_call_fn_161837_;?8
1?.
(?%
inputs?????????	c0
p
? " ??????????	c0?
+__inference_dropout_55_layer_call_fn_161842_;?8
1?.
(?%
inputs?????????	c0
p 
? " ??????????	c0?
F__inference_flatten_27_layer_call_and_return_conditional_losses_161848b7?4
-?*
(?%
inputs?????????	c0
? "'?$
?
0???????????
? ?
+__inference_flatten_27_layer_call_fn_161853U7?4
-?*
(?%
inputs?????????	c0
? "????????????;
__inference_loss_fn_0_161886?

? 
? "? ;
__inference_loss_fn_1_161899?

? 
? "? ;
__inference_loss_fn_2_161912?

? 
? "? ;
__inference_loss_fn_3_161925?

? 
? "? ?
L__inference_max_pooling2d_54_layer_call_and_return_conditional_losses_161114?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_54_layer_call_fn_161120?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_161164?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_55_layer_call_fn_161170?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
I__inference_sequential_87_layer_call_and_return_conditional_losses_161320z/0I?F
??<
2?/
conv2d_54_input?????????(?
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161376z/0I?F
??<
2?/
conv2d_54_input?????????(?
p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161659q/0@?=
6?3
)?&
inputs?????????(?
p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_87_layer_call_and_return_conditional_losses_161722q/0@?=
6?3
)?&
inputs?????????(?
p 

 
? "%?"
?
0?????????
? ?
.__inference_sequential_87_layer_call_fn_161450m/0I?F
??<
2?/
conv2d_54_input?????????(?
p

 
? "???????????
.__inference_sequential_87_layer_call_fn_161523m/0I?F
??<
2?/
conv2d_54_input?????????(?
p 

 
? "???????????
.__inference_sequential_87_layer_call_fn_161739d/0@?=
6?3
)?&
inputs?????????(?
p

 
? "???????????
.__inference_sequential_87_layer_call_fn_161756d/0@?=
6?3
)?&
inputs?????????(?
p 

 
? "???????????
$__inference_signature_wrapper_161582?/0T?Q
? 
J?G
E
conv2d_54_input2?/
conv2d_54_input?????????(?"3?0
.
dense_51"?
dense_51?????????