КС
й
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ио
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
Ќ
(QRnnNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*9
shared_name*(QRnnNetwork/EncodingNetwork/dense/kernel
Ѕ
<QRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp(QRnnNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:*
dtype0
Є
&QRnnNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&QRnnNetwork/EncodingNetwork/dense/bias

:QRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp&QRnnNetwork/EncodingNetwork/dense/bias*
_output_shapes
:*
dtype0
А
*QRnnNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:f(*;
shared_name,*QRnnNetwork/EncodingNetwork/dense_1/kernel
Љ
>QRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp*QRnnNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes

:f(*
dtype0
Ј
(QRnnNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*9
shared_name*(QRnnNetwork/EncodingNetwork/dense_1/bias
Ё
<QRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp(QRnnNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
:(*
dtype0

!QRnnNetwork/dynamic_unroll/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:(P*2
shared_name#!QRnnNetwork/dynamic_unroll/kernel

5QRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpReadVariableOp!QRnnNetwork/dynamic_unroll/kernel*
_output_shapes

:(P*
dtype0
В
+QRnnNetwork/dynamic_unroll/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*<
shared_name-+QRnnNetwork/dynamic_unroll/recurrent_kernel
Ћ
?QRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpReadVariableOp+QRnnNetwork/dynamic_unroll/recurrent_kernel*
_output_shapes

:P*
dtype0

QRnnNetwork/dynamic_unroll/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*0
shared_name!QRnnNetwork/dynamic_unroll/bias

3QRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpReadVariableOpQRnnNetwork/dynamic_unroll/bias*
_output_shapes
:P*
dtype0

QRnnNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameQRnnNetwork/dense_2/kernel

.QRnnNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOpQRnnNetwork/dense_2/kernel*
_output_shapes

:*
dtype0

QRnnNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameQRnnNetwork/dense_2/bias

,QRnnNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOpQRnnNetwork/dense_2/bias*
_output_shapes
:*
dtype0
В
+QRnnNetwork/num_action_project/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+QRnnNetwork/num_action_project/dense/kernel
Ћ
?QRnnNetwork/num_action_project/dense/kernel/Read/ReadVariableOpReadVariableOp+QRnnNetwork/num_action_project/dense/kernel*
_output_shapes

:*
dtype0
Њ
)QRnnNetwork/num_action_project/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)QRnnNetwork/num_action_project/dense/bias
Ѓ
=QRnnNetwork/num_action_project/dense/bias/Read/ReadVariableOpReadVariableOp)QRnnNetwork/num_action_project/dense/bias*
_output_shapes
:*
dtype0

NoOpNoOp
љ-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Д-
valueЊ-BЇ- B -

collect_data_spec
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures

observation
1
 
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
N
	0

1
2
3
4
5
6
7
8
9
10
#
0
1
2
3
4
 
 
jh
VARIABLE_VALUE(QRnnNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QRnnNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE*QRnnNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QRnnNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE!QRnnNetwork/dynamic_unroll/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE+QRnnNetwork/dynamic_unroll/recurrent_kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEQRnnNetwork/dynamic_unroll/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEQRnnNetwork/dense_2/kernel,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEQRnnNetwork/dense_2/bias,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE+QRnnNetwork/num_action_project/dense/kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE)QRnnNetwork/num_action_project/dense/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE

ref
1

ref
1

ref
1

ref
1

ref
1

observation
3
 

	state
1

observation
1
j

_q_network
_time_step_spec
_policy_state_spec
_policy_step_spec
 _trajectory_spec
З
!_input_tensor_spec
_state_spec
"_input_encoder
#_lstm_network
$_output_encoder
%regularization_losses
&	variables
'trainable_variables
(	keras_api

	state
1

observation
1
 
м
)_input_tensor_spec
*_preprocessing_nest
+_flat_preprocessing_layers
,_preprocessing_combiner
-_postprocessing_layers
.regularization_losses
/	variables
0trainable_variables
1	keras_api
\
2cell
3regularization_losses
4	variables
5trainable_variables
6	keras_api

70
81
 
N
	0

1
2
3
4
5
6
7
8
9
10
N
	0

1
2
3
4
5
6
7
8
9
10
­
%regularization_losses
9layer_metrics
:non_trainable_variables

;layers
&	variables
'trainable_variables
<layer_regularization_losses
=metrics
 
 

>0
?1
R
@regularization_losses
A	variables
Btrainable_variables
C	keras_api

D0
E1
 

	0

1
2
3

	0

1
2
3
­
.regularization_losses
Flayer_metrics
Gnon_trainable_variables

Hlayers
/	variables
0trainable_variables
Ilayer_regularization_losses
Jmetrics
~

kernel
recurrent_kernel
bias
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
 

0
1
2

0
1
2
­
3regularization_losses
Olayer_metrics
Pnon_trainable_variables

Qlayers
4	variables
5trainable_variables
Rlayer_regularization_losses
Smetrics
h

kernel
bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
h

kernel
bias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
 
 

"0
#1
72
83
 
 
h

	kernel

bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
R
`regularization_losses
a	variables
btrainable_variables
c	keras_api
 
 
 
­
@regularization_losses
dlayer_metrics
enon_trainable_variables

flayers
A	variables
Btrainable_variables
glayer_regularization_losses
hmetrics
R
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
h

kernel
bias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
 
 
#
>0
?1
,2
D3
E4
 
 
 

0
1
2

0
1
2
­
Kregularization_losses
qlayer_metrics
rnon_trainable_variables

slayers
L	variables
Mtrainable_variables
tlayer_regularization_losses
umetrics
 
 

20
 
 
 

0
1

0
1
­
Tregularization_losses
vlayer_metrics
wnon_trainable_variables

xlayers
U	variables
Vtrainable_variables
ylayer_regularization_losses
zmetrics
 

0
1

0
1
­
Xregularization_losses
{layer_metrics
|non_trainable_variables

}layers
Y	variables
Ztrainable_variables
~layer_regularization_losses
metrics
 

	0

1

	0

1
В
\regularization_losses
layer_metrics
non_trainable_variables
layers
]	variables
^trainable_variables
 layer_regularization_losses
metrics
 
 
 
В
`regularization_losses
layer_metrics
non_trainable_variables
layers
a	variables
btrainable_variables
 layer_regularization_losses
metrics
 
 
 
 
 
 
 
 
В
iregularization_losses
layer_metrics
non_trainable_variables
layers
j	variables
ktrainable_variables
 layer_regularization_losses
metrics
 

0
1

0
1
В
mregularization_losses
layer_metrics
non_trainable_variables
layers
n	variables
otrainable_variables
 layer_regularization_losses
metrics
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
l
action_0/discountPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
{
action_0/observation/posPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

action_0/observation/pricePlaceholder*+
_output_shapes
:џџџџџџџџџ*
dtype0* 
shape:џџџџџџџџџ
j
action_0/rewardPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m
action_0/step_typePlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m

action_1/0Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m

action_1/1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observation/posaction_0/observation/priceaction_0/rewardaction_0/step_type
action_1/0
action_1/1(QRnnNetwork/EncodingNetwork/dense/kernel&QRnnNetwork/EncodingNetwork/dense/bias*QRnnNetwork/EncodingNetwork/dense_1/kernel(QRnnNetwork/EncodingNetwork/dense_1/bias!QRnnNetwork/dynamic_unroll/kernel+QRnnNetwork/dynamic_unroll/recurrent_kernelQRnnNetwork/dynamic_unroll/biasQRnnNetwork/dense_2/kernelQRnnNetwork/dense_2/bias+QRnnNetwork/num_action_project/dense/kernel)QRnnNetwork/num_action_project/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1033060305
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Н
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1033060314
о
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1033060326

StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *1
f,R*
(__inference_signature_wrapper_1033060322
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ь
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp<QRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOp:QRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOp>QRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOp<QRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOp5QRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOp?QRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOp3QRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOp.QRnnNetwork/dense_2/kernel/Read/ReadVariableOp,QRnnNetwork/dense_2/bias/Read/ReadVariableOp?QRnnNetwork/num_action_project/dense/kernel/Read/ReadVariableOp=QRnnNetwork/num_action_project/dense/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_save_1033060397
з
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable(QRnnNetwork/EncodingNetwork/dense/kernel&QRnnNetwork/EncodingNetwork/dense/bias*QRnnNetwork/EncodingNetwork/dense_1/kernel(QRnnNetwork/EncodingNetwork/dense_1/bias!QRnnNetwork/dynamic_unroll/kernel+QRnnNetwork/dynamic_unroll/recurrent_kernelQRnnNetwork/dynamic_unroll/biasQRnnNetwork/dense_2/kernelQRnnNetwork/dense_2/bias+QRnnNetwork/num_action_project/dense/kernel)QRnnNetwork/num_action_project/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference__traced_restore_1033060443Ж№
/

__inference_<lambda>_993*
_input_shapes 
Ћѕ

__inference_action_17167228
time_step_step_type
time_step_reward
time_step_discount
time_step_observation_pos
time_step_observation_price
policy_state_0
policy_state_1D
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceE
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceF
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceG
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceG
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resourceI
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resourceH
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource9
5qrnnnetwork_dense_2_tensordot_readvariableop_resource7
3qrnnnetwork_dense_2_biasadd_readvariableop_resourceJ
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resourceH
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource
identity

identity_1

identity_2P
ShapeShapetime_step_discount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yl
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state_0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_1T
Shape_2Shapetime_step_discount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yr
Equal_1Equaltime_step_step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_2

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_3z
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims/dimД
QRnnNetwork/ExpandDims
ExpandDimstime_step_observation_pos#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims~
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_1/dimР
QRnnNetwork/ExpandDims_1
ExpandDimstime_step_observation_price%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_1~
QRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_2/dimА
QRnnNetwork/ExpandDims_2
ExpandDimstime_step_step_type%QRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_2С
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	21
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeУ
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeЧ
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShape!QRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeЫ
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2;
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape
3QRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshape!QRnnNetwork/ExpandDims_1:output:0BQRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ25
3QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshapeе
&QRnnNetwork/EncodingNetwork/dense/CastCast:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/EncodingNetwork/dense/Castѓ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp§
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul*QRnnNetwork/EncodingNetwork/dense/Cast:y:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/EncodingNetwork/dense/MatMulђ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/EncodingNetwork/dense/BiasAddЇ
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2+
)QRnnNetwork/EncodingNetwork/flatten/Const
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape<QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2-
+QRnnNetwork/EncodingNetwork/flatten/ReshapeЌ
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :25
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisЯ
.QRnnNetwork/EncodingNetwork/concatenate/concatConcatV22QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:04QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0<QRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџf20
.QRnnNetwork/EncodingNetwork/concatenate/concatЋ
+QRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   2-
+QRnnNetwork/EncodingNetwork/flatten_1/Const
-QRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshape7QRnnNetwork/EncodingNetwork/concatenate/concat:output:04QRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf2/
-QRnnNetwork/EncodingNetwork/flatten_1/Reshapeљ
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:f(*
dtype02;
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul6QRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2,
*QRnnNetwork/EncodingNetwork/dense_1/MatMulј
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02<
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2-
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddФ
(QRnnNetwork/EncodingNetwork/dense_1/ReluRelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(2*
(QRnnNetwork/EncodingNetwork/dense_1/ReluЬ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1а
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ш
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice:QRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2;
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceм
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_unflatten/Shapeа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackд
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1д
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2№
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2=
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1Д
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisю
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:24
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatЎ
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ(25
3QRnnNetwork/EncodingNetwork/batch_unflatten/Reshapej
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2
QRnnNetwork/mask/y
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_2:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/mask
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2!
QRnnNetwork/dynamic_unroll/Rank
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/start
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/deltaѕ
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/rangeЉ
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2,
*QRnnNetwork/dynamic_unroll/concat/values_0
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&QRnnNetwork/dynamic_unroll/concat/axis
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!QRnnNetwork/dynamic_unroll/concatљ
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(2&
$QRnnNetwork/dynamic_unroll/transpose
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/ShapeЊ
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.QRnnNetwork/dynamic_unroll/strided_slice/stackЎ
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Ў
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(QRnnNetwork/dynamic_unroll/strided_sliceЋ
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2-
+QRnnNetwork/dynamic_unroll/transpose_1/permл
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/dynamic_unroll/transpose_1
&QRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/zeros/mul/yи
$QRnnNetwork/dynamic_unroll/zeros/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:0/QRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dynamic_unroll/zeros/mul
'QRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2)
'QRnnNetwork/dynamic_unroll/zeros/Less/yг
%QRnnNetwork/dynamic_unroll/zeros/LessLess(QRnnNetwork/dynamic_unroll/zeros/mul:z:00QRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%QRnnNetwork/dynamic_unroll/zeros/Less
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/zeros/packed/1я
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'QRnnNetwork/dynamic_unroll/zeros/packed
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&QRnnNetwork/dynamic_unroll/zeros/Constс
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 QRnnNetwork/dynamic_unroll/zeros
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yо
&QRnnNetwork/dynamic_unroll/zeros_1/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:01QRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2(
&QRnnNetwork/dynamic_unroll/zeros_1/mul
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2+
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yл
'QRnnNetwork/dynamic_unroll/zeros_1/LessLess*QRnnNetwork/dynamic_unroll/zeros_1/mul:z:02QRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2)
'QRnnNetwork/dynamic_unroll/zeros_1/Less
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1ѕ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)QRnnNetwork/dynamic_unroll/zeros_1/packed
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(QRnnNetwork/dynamic_unroll/zeros_1/Constщ
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"QRnnNetwork/dynamic_unroll/zeros_1Ц
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ(*
squeeze_dims
 2$
"QRnnNetwork/dynamic_unroll/SqueezeШ
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2&
$QRnnNetwork/dynamic_unroll/Squeeze_1ё
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!QRnnNetwork/dynamic_unroll/Selectї
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#QRnnNetwork/dynamic_unroll/Select_1ќ
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:(P*
dtype02<
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2-
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMul
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02>
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2/
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1џ
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP2*
(QRnnNetwork/dynamic_unroll/lstm_cell/addћ
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02=
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2.
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2,
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstЎ
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimг
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*QRnnNetwork/dynamic_unroll/lstm_cell/splitЮ
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoidв
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1я
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/dynamic_unroll/lstm_cell/mulХ
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/dynamic_unroll/lstm_cell/Tanhђ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1ё
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1в
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Ф
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2-
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1і
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/ExpandDims/dimі
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2'
%QRnnNetwork/dynamic_unroll/ExpandDimsв
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02.
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"QRnnNetwork/dense_2/Tensordot/axes
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"QRnnNetwork/dense_2/Tensordot/freeЈ
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/Shape
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisЕ
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/GatherV2 
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisЛ
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(QRnnNetwork/dense_2/Tensordot/GatherV2_1
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#QRnnNetwork/dense_2/Tensordot/Constа
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"QRnnNetwork/dense_2/Tensordot/Prod
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%QRnnNetwork/dense_2/Tensordot/Const_1и
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dense_2/Tensordot/Prod_1
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)QRnnNetwork/dense_2/Tensordot/concat/axis
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$QRnnNetwork/dense_2/Tensordot/concatм
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/stackє
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2)
'QRnnNetwork/dense_2/Tensordot/transposeя
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2'
%QRnnNetwork/dense_2/Tensordot/Reshapeю
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$QRnnNetwork/dense_2/Tensordot/MatMul
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%QRnnNetwork/dense_2/Tensordot/Const_2
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/concat_1/axisЁ
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/concat_1р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/TensordotШ
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpз
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/BiasAdd
QRnnNetwork/dense_2/ReluRelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/Relu
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02?
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpД
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3QRnnNetwork/num_action_project/dense/Tensordot/axesЛ
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3QRnnNetwork/num_action_project/dense/Tensordot/freeТ
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeО
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2Т
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1Ж
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4QRnnNetwork/num_action_project/dense/Tensordot/Const
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3QRnnNetwork/num_action_project/dense/Tensordot/ProdК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1К
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisщ
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5QRnnNetwork/num_action_project/dense/Tensordot/concat 
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/stack
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Relu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2:
8QRnnNetwork/num_action_project/dense/Tensordot/transposeГ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeВ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ27
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2О
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisі
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1Є
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/num_action_project/dense/Tensordotћ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/num_action_project/dense/BiasAddЕ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
QRnnNetwork/Squeeze
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionК
Categorical_1/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xД
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2Щ
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsЯ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЊ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЮ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1а
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/yВ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*О
_input_shapesЌ
Љ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::::::X T
#
_output_shapes
:џџџџџџџџџ
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:џџџџџџџџџ
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:џџџџџџџџџ
,
_user_specified_nametime_step/discount:b^
'
_output_shapes
:џџџџџџџџџ
3
_user_specified_nametime_step/observation/pos:hd
+
_output_shapes
:џџџџџџџџџ
5
_user_specified_nametime_step/observation/price:WS
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namepolicy_state/0:WS
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_namepolicy_state/1
8

&__inference__traced_restore_1033060443
file_prefix
assignvariableop_variable?
;assignvariableop_1_qrnnnetwork_encodingnetwork_dense_kernel=
9assignvariableop_2_qrnnnetwork_encodingnetwork_dense_biasA
=assignvariableop_3_qrnnnetwork_encodingnetwork_dense_1_kernel?
;assignvariableop_4_qrnnnetwork_encodingnetwork_dense_1_bias8
4assignvariableop_5_qrnnnetwork_dynamic_unroll_kernelB
>assignvariableop_6_qrnnnetwork_dynamic_unroll_recurrent_kernel6
2assignvariableop_7_qrnnnetwork_dynamic_unroll_bias1
-assignvariableop_8_qrnnnetwork_dense_2_kernel/
+assignvariableop_9_qrnnnetwork_dense_2_biasC
?assignvariableop_10_qrnnnetwork_num_action_project_dense_kernelA
=assignvariableop_11_qrnnnetwork_num_action_project_dense_bias
identity_13ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Щ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*е
valueЫBШB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesь
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Р
AssignVariableOp_1AssignVariableOp;assignvariableop_1_qrnnnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2О
AssignVariableOp_2AssignVariableOp9assignvariableop_2_qrnnnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Т
AssignVariableOp_3AssignVariableOp=assignvariableop_3_qrnnnetwork_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Р
AssignVariableOp_4AssignVariableOp;assignvariableop_4_qrnnnetwork_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Й
AssignVariableOp_5AssignVariableOp4assignvariableop_5_qrnnnetwork_dynamic_unroll_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6У
AssignVariableOp_6AssignVariableOp>assignvariableop_6_qrnnnetwork_dynamic_unroll_recurrent_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7З
AssignVariableOp_7AssignVariableOp2assignvariableop_7_qrnnnetwork_dynamic_unroll_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8В
AssignVariableOp_8AssignVariableOp-assignvariableop_8_qrnnnetwork_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9А
AssignVariableOp_9AssignVariableOp+assignvariableop_9_qrnnnetwork_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ч
AssignVariableOp_10AssignVariableOp?assignvariableop_10_qrnnnetwork_num_action_project_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Х
AssignVariableOp_11AssignVariableOp=assignvariableop_11_qrnnnetwork_num_action_project_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpц
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12й
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

b
(__inference_signature_wrapper_1033060322
unknown
identity	ЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_171669472
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall

Q
!__inference_get_initial_state_984

batch_size
identity

identity_1R
packedPack
batch_size*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1b
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Ы
*
(__inference_signature_wrapper_1033060326
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_171669582
PartitionedCall*
_input_shapes 

X
(__inference_signature_wrapper_1033060314

batch_size
identity

identity_1Т
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_171669312
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityp

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Я
H
__inference_<lambda>_990
readvariableop_resource
identity	p
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpY
IdentityIdentityReadVariableOp:value:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
ще
Ъ
 __inference_distribution_fn_1505
	step_type

reward
discount
observation_pos
observation_price
unknown
	unknown_0D
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceE
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceF
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceG
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceG
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resourceI
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resourceH
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource9
5qrnnnetwork_dense_2_tensordot_readvariableop_resource7
3qrnnnetwork_dense_2_biasadd_readvariableop_resourceJ
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resourceH
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource
identity

identity_1

identity_2F
ShapeShapediscount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape}
SelectV2SelectV2Reshape:output:0zeros:output:0unknown*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0	unknown_0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_1J
Shape_2Shapediscount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_2

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_3z
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims/dimЊ
QRnnNetwork/ExpandDims
ExpandDimsobservation_pos#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims~
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_1/dimЖ
QRnnNetwork/ExpandDims_1
ExpandDimsobservation_price%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_1~
QRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_2/dimІ
QRnnNetwork/ExpandDims_2
ExpandDims	step_type%QRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_2С
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	21
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeУ
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeЧ
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShape!QRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeЫ
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2;
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape
3QRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshape!QRnnNetwork/ExpandDims_1:output:0BQRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ25
3QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshapeе
&QRnnNetwork/EncodingNetwork/dense/CastCast:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/EncodingNetwork/dense/Castѓ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp§
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul*QRnnNetwork/EncodingNetwork/dense/Cast:y:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/EncodingNetwork/dense/MatMulђ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/EncodingNetwork/dense/BiasAddЇ
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2+
)QRnnNetwork/EncodingNetwork/flatten/Const
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape<QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2-
+QRnnNetwork/EncodingNetwork/flatten/ReshapeЌ
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :25
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisЯ
.QRnnNetwork/EncodingNetwork/concatenate/concatConcatV22QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:04QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0<QRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџf20
.QRnnNetwork/EncodingNetwork/concatenate/concatЋ
+QRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   2-
+QRnnNetwork/EncodingNetwork/flatten_1/Const
-QRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshape7QRnnNetwork/EncodingNetwork/concatenate/concat:output:04QRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf2/
-QRnnNetwork/EncodingNetwork/flatten_1/Reshapeљ
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:f(*
dtype02;
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul6QRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2,
*QRnnNetwork/EncodingNetwork/dense_1/MatMulј
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02<
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2-
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddФ
(QRnnNetwork/EncodingNetwork/dense_1/ReluRelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(2*
(QRnnNetwork/EncodingNetwork/dense_1/ReluЬ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1а
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ш
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice:QRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2;
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceм
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_unflatten/Shapeа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackд
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1д
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2№
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2=
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1Д
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisю
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:24
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatЎ
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ(25
3QRnnNetwork/EncodingNetwork/batch_unflatten/Reshapej
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2
QRnnNetwork/mask/y
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_2:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/mask
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2!
QRnnNetwork/dynamic_unroll/Rank
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/start
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/deltaѕ
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/rangeЉ
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2,
*QRnnNetwork/dynamic_unroll/concat/values_0
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&QRnnNetwork/dynamic_unroll/concat/axis
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!QRnnNetwork/dynamic_unroll/concatљ
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(2&
$QRnnNetwork/dynamic_unroll/transpose
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/ShapeЊ
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.QRnnNetwork/dynamic_unroll/strided_slice/stackЎ
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Ў
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(QRnnNetwork/dynamic_unroll/strided_sliceЋ
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2-
+QRnnNetwork/dynamic_unroll/transpose_1/permл
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/dynamic_unroll/transpose_1
&QRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/zeros/mul/yи
$QRnnNetwork/dynamic_unroll/zeros/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:0/QRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dynamic_unroll/zeros/mul
'QRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2)
'QRnnNetwork/dynamic_unroll/zeros/Less/yг
%QRnnNetwork/dynamic_unroll/zeros/LessLess(QRnnNetwork/dynamic_unroll/zeros/mul:z:00QRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%QRnnNetwork/dynamic_unroll/zeros/Less
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/zeros/packed/1я
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'QRnnNetwork/dynamic_unroll/zeros/packed
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&QRnnNetwork/dynamic_unroll/zeros/Constс
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 QRnnNetwork/dynamic_unroll/zeros
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yо
&QRnnNetwork/dynamic_unroll/zeros_1/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:01QRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2(
&QRnnNetwork/dynamic_unroll/zeros_1/mul
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2+
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yл
'QRnnNetwork/dynamic_unroll/zeros_1/LessLess*QRnnNetwork/dynamic_unroll/zeros_1/mul:z:02QRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2)
'QRnnNetwork/dynamic_unroll/zeros_1/Less
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1ѕ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)QRnnNetwork/dynamic_unroll/zeros_1/packed
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(QRnnNetwork/dynamic_unroll/zeros_1/Constщ
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"QRnnNetwork/dynamic_unroll/zeros_1Ц
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ(*
squeeze_dims
 2$
"QRnnNetwork/dynamic_unroll/SqueezeШ
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2&
$QRnnNetwork/dynamic_unroll/Squeeze_1ё
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!QRnnNetwork/dynamic_unroll/Selectї
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#QRnnNetwork/dynamic_unroll/Select_1ќ
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:(P*
dtype02<
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2-
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMul
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02>
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2/
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1џ
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP2*
(QRnnNetwork/dynamic_unroll/lstm_cell/addћ
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02=
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2.
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2,
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstЎ
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimг
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*QRnnNetwork/dynamic_unroll/lstm_cell/splitЮ
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoidв
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1я
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/dynamic_unroll/lstm_cell/mulХ
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/dynamic_unroll/lstm_cell/Tanhђ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1ё
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1в
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Ф
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2-
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1і
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/ExpandDims/dimі
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2'
%QRnnNetwork/dynamic_unroll/ExpandDimsв
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02.
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"QRnnNetwork/dense_2/Tensordot/axes
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"QRnnNetwork/dense_2/Tensordot/freeЈ
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/Shape
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisЕ
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/GatherV2 
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisЛ
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(QRnnNetwork/dense_2/Tensordot/GatherV2_1
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#QRnnNetwork/dense_2/Tensordot/Constа
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"QRnnNetwork/dense_2/Tensordot/Prod
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%QRnnNetwork/dense_2/Tensordot/Const_1и
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dense_2/Tensordot/Prod_1
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)QRnnNetwork/dense_2/Tensordot/concat/axis
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$QRnnNetwork/dense_2/Tensordot/concatм
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/stackє
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2)
'QRnnNetwork/dense_2/Tensordot/transposeя
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2'
%QRnnNetwork/dense_2/Tensordot/Reshapeю
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$QRnnNetwork/dense_2/Tensordot/MatMul
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%QRnnNetwork/dense_2/Tensordot/Const_2
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/concat_1/axisЁ
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/concat_1р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/TensordotШ
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpз
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/BiasAdd
QRnnNetwork/dense_2/ReluRelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/Relu
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02?
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpД
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3QRnnNetwork/num_action_project/dense/Tensordot/axesЛ
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3QRnnNetwork/num_action_project/dense/Tensordot/freeТ
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeО
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2Т
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1Ж
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4QRnnNetwork/num_action_project/dense/Tensordot/Const
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3QRnnNetwork/num_action_project/dense/Tensordot/ProdК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1К
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisщ
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5QRnnNetwork/num_action_project/dense/Tensordot/concat 
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/stack
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Relu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2:
8QRnnNetwork/num_action_project/dense/Tensordot/transposeГ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeВ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ27
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2О
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisі
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1Є
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/num_action_project/dense/Tensordotћ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/num_action_project/dense/BiasAddЕ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
QRnnNetwork/Squeeze
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionК
Categorical_1/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtolk
IdentityIdentityCategorical_1/mode/Cast:y:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2n
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*О
_input_shapesЌ
Љ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::::::N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameobservation/pos:^Z
+
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameobservation/price:JF
'
_output_shapes
:џџџџџџџџџ

_user_specified_name0:JF
'
_output_shapes
:џџџџџџџџџ

_user_specified_name1
ї
f
,__inference_function_with_signature_17166947
unknown
identity	ЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_<lambda>_9902
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
щѓ
Э
__inference_action_17166839
	time_step
time_step_1
time_step_2
time_step_3
time_step_4
policy_state
policy_state_1D
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceE
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceF
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceG
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceG
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resourceI
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resourceH
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource9
5qrnnnetwork_dense_2_tensordot_readvariableop_resource7
3qrnnnetwork_dense_2_biasadd_readvariableop_resourceJ
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resourceH
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource
identity

identity_1

identity_2I
ShapeShapetime_step_2*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape
SelectV2SelectV2Reshape:output:0zeros:output:0policy_state*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0policy_state_1*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_1M
Shape_2Shapetime_step_2*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	time_stepEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_2

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_3z
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims/dimІ
QRnnNetwork/ExpandDims
ExpandDimstime_step_3#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims~
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_1/dimА
QRnnNetwork/ExpandDims_1
ExpandDimstime_step_4%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_1~
QRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_2/dimІ
QRnnNetwork/ExpandDims_2
ExpandDims	time_step%QRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_2С
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	21
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeУ
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeЧ
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShape!QRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeЫ
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2;
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape
3QRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshape!QRnnNetwork/ExpandDims_1:output:0BQRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ25
3QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshapeе
&QRnnNetwork/EncodingNetwork/dense/CastCast:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/EncodingNetwork/dense/Castѓ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp§
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul*QRnnNetwork/EncodingNetwork/dense/Cast:y:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/EncodingNetwork/dense/MatMulђ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/EncodingNetwork/dense/BiasAddЇ
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2+
)QRnnNetwork/EncodingNetwork/flatten/Const
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape<QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2-
+QRnnNetwork/EncodingNetwork/flatten/ReshapeЌ
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :25
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisЯ
.QRnnNetwork/EncodingNetwork/concatenate/concatConcatV22QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:04QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0<QRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџf20
.QRnnNetwork/EncodingNetwork/concatenate/concatЋ
+QRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   2-
+QRnnNetwork/EncodingNetwork/flatten_1/Const
-QRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshape7QRnnNetwork/EncodingNetwork/concatenate/concat:output:04QRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf2/
-QRnnNetwork/EncodingNetwork/flatten_1/Reshapeљ
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:f(*
dtype02;
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul6QRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2,
*QRnnNetwork/EncodingNetwork/dense_1/MatMulј
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02<
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2-
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddФ
(QRnnNetwork/EncodingNetwork/dense_1/ReluRelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(2*
(QRnnNetwork/EncodingNetwork/dense_1/ReluЬ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1а
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ш
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice:QRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2;
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceм
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_unflatten/Shapeа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackд
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1д
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2№
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2=
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1Д
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisю
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:24
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatЎ
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ(25
3QRnnNetwork/EncodingNetwork/batch_unflatten/Reshapej
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2
QRnnNetwork/mask/y
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_2:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/mask
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2!
QRnnNetwork/dynamic_unroll/Rank
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/start
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/deltaѕ
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/rangeЉ
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2,
*QRnnNetwork/dynamic_unroll/concat/values_0
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&QRnnNetwork/dynamic_unroll/concat/axis
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!QRnnNetwork/dynamic_unroll/concatљ
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(2&
$QRnnNetwork/dynamic_unroll/transpose
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/ShapeЊ
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.QRnnNetwork/dynamic_unroll/strided_slice/stackЎ
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Ў
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(QRnnNetwork/dynamic_unroll/strided_sliceЋ
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2-
+QRnnNetwork/dynamic_unroll/transpose_1/permл
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/dynamic_unroll/transpose_1
&QRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/zeros/mul/yи
$QRnnNetwork/dynamic_unroll/zeros/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:0/QRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dynamic_unroll/zeros/mul
'QRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2)
'QRnnNetwork/dynamic_unroll/zeros/Less/yг
%QRnnNetwork/dynamic_unroll/zeros/LessLess(QRnnNetwork/dynamic_unroll/zeros/mul:z:00QRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%QRnnNetwork/dynamic_unroll/zeros/Less
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/zeros/packed/1я
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'QRnnNetwork/dynamic_unroll/zeros/packed
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&QRnnNetwork/dynamic_unroll/zeros/Constс
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 QRnnNetwork/dynamic_unroll/zeros
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yо
&QRnnNetwork/dynamic_unroll/zeros_1/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:01QRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2(
&QRnnNetwork/dynamic_unroll/zeros_1/mul
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2+
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yл
'QRnnNetwork/dynamic_unroll/zeros_1/LessLess*QRnnNetwork/dynamic_unroll/zeros_1/mul:z:02QRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2)
'QRnnNetwork/dynamic_unroll/zeros_1/Less
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1ѕ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)QRnnNetwork/dynamic_unroll/zeros_1/packed
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(QRnnNetwork/dynamic_unroll/zeros_1/Constщ
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"QRnnNetwork/dynamic_unroll/zeros_1Ц
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ(*
squeeze_dims
 2$
"QRnnNetwork/dynamic_unroll/SqueezeШ
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2&
$QRnnNetwork/dynamic_unroll/Squeeze_1ё
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!QRnnNetwork/dynamic_unroll/Selectї
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#QRnnNetwork/dynamic_unroll/Select_1ќ
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:(P*
dtype02<
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2-
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMul
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02>
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2/
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1џ
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP2*
(QRnnNetwork/dynamic_unroll/lstm_cell/addћ
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02=
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2.
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2,
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstЎ
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimг
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*QRnnNetwork/dynamic_unroll/lstm_cell/splitЮ
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoidв
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1я
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/dynamic_unroll/lstm_cell/mulХ
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/dynamic_unroll/lstm_cell/Tanhђ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1ё
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1в
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Ф
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2-
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1і
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/ExpandDims/dimі
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2'
%QRnnNetwork/dynamic_unroll/ExpandDimsв
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02.
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"QRnnNetwork/dense_2/Tensordot/axes
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"QRnnNetwork/dense_2/Tensordot/freeЈ
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/Shape
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisЕ
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/GatherV2 
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisЛ
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(QRnnNetwork/dense_2/Tensordot/GatherV2_1
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#QRnnNetwork/dense_2/Tensordot/Constа
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"QRnnNetwork/dense_2/Tensordot/Prod
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%QRnnNetwork/dense_2/Tensordot/Const_1и
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dense_2/Tensordot/Prod_1
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)QRnnNetwork/dense_2/Tensordot/concat/axis
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$QRnnNetwork/dense_2/Tensordot/concatм
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/stackє
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2)
'QRnnNetwork/dense_2/Tensordot/transposeя
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2'
%QRnnNetwork/dense_2/Tensordot/Reshapeю
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$QRnnNetwork/dense_2/Tensordot/MatMul
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%QRnnNetwork/dense_2/Tensordot/Const_2
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/concat_1/axisЁ
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/concat_1р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/TensordotШ
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpз
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/BiasAdd
QRnnNetwork/dense_2/ReluRelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/Relu
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02?
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpД
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3QRnnNetwork/num_action_project/dense/Tensordot/axesЛ
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3QRnnNetwork/num_action_project/dense/Tensordot/freeТ
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeО
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2Т
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1Ж
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4QRnnNetwork/num_action_project/dense/Tensordot/Const
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3QRnnNetwork/num_action_project/dense/Tensordot/ProdК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1К
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisщ
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5QRnnNetwork/num_action_project/dense/Tensordot/concat 
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/stack
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Relu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2:
8QRnnNetwork/num_action_project/dense/Tensordot/transposeГ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeВ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ27
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2О
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisі
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1Є
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/num_action_project/dense/Tensordotћ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/num_action_project/dense/BiasAddЕ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
QRnnNetwork/Squeeze
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionК
Categorical_1/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xД
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2Щ
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsЯ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЊ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЮ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1а
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/yВ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*О
_input_shapesЌ
Љ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::::::N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:VR
+
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:UQ
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state:UQ
'
_output_shapes
:џџџџџџџџџ
&
_user_specified_namepolicy_state
М

(__inference_signature_wrapper_1033060305
discount
observation_pos
observation_price

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity

identity_1

identity_2ЂStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_posobservation_priceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *5
f0R.
,__inference_function_with_signature_171668682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*О
_input_shapesЌ
Љ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:ZV
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_name0/observation/pos:`\
+
_output_shapes
:џџџџџџџџџ
-
_user_specified_name0/observation/price:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:LH
'
_output_shapes
:џџџџџџџџџ

_user_specified_name1/0:LH
'
_output_shapes
:џџџџџџџџџ

_user_specified_name1/1
Џ

,__inference_function_with_signature_17166868
	step_type

reward
discount
observation_pos
observation_price
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity

identity_1

identity_2ЂStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservation_posobservation_priceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *$
fR
__inference_action_171668392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*О
_input_shapesЌ
Љ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:OK
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:ZV
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_name0/observation/pos:`\
+
_output_shapes
:џџџџџџџџџ
-
_user_specified_name0/observation/price:LH
'
_output_shapes
:џџџџџџџџџ

_user_specified_name1/0:LH
'
_output_shapes
:џџџџџџџџџ

_user_specified_name1/1

V
&__inference_get_initial_state_17166926

batch_size
identity

identity_1R
packedPack
batch_size*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1b
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Л
.
,__inference_function_with_signature_17166958ѕ
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *!
fR
__inference_<lambda>_9932
PartitionedCall*
_input_shapes 
Ъѓ
С
__inference_action_1259
	step_type

reward
discount
observation_pos
observation_price
unknown
	unknown_0D
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resourceE
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceF
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceG
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resourceG
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resourceI
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resourceH
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource9
5qrnnnetwork_dense_2_tensordot_readvariableop_resource7
3qrnnnetwork_dense_2_biasadd_readvariableop_resourceJ
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resourceH
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource
identity

identity_1

identity_2F
ShapeShapediscount*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2т
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice^
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
packedl
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Consto
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
zerosp
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_1`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1c
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Constw
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_1T
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
Equal/yb
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
EqualN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1R
subSubRank:output:0Rank_1:output:0*
T0*
_output_shapes
: 2
subK
Shape_1Shape	Equal:z:0*
T0
*
_output_shapes
:2	
Shape_1{
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones/Reshape/shaper
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones/ReshapeZ

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

ones/Conste
onesFillones/Reshape:output:0ones/Const:output:0*
T0*
_output_shapes
:2
ones`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_2/axis
concat_2ConcatV2Shape_1:output:0ones:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:2

concat_2m
ReshapeReshape	Equal:z:0concat_2:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2	
Reshape}
SelectV2SelectV2Reshape:output:0zeros:output:0unknown*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2

SelectV2_1SelectV2Reshape:output:0zeros_1:output:0	unknown_0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_1J
Shape_2Shapediscount*
T0*
_output_shapes
:2	
Shape_2x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ю
strided_slice_1StridedSliceShape_2:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1d
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:2

packed_1p
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_3/axis
concat_3ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:2

concat_3c
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_2/Constw
zeros_2Fillconcat_3:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_2p
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:2
shape_as_tensor_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_4/axis
concat_4ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_4/axis:output:0*
N*
T0*
_output_shapes
:2

concat_4c
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_3/Constw
zeros_3Fillconcat_4:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
zeros_3X
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
	Equal_1/yh
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2	
Equal_1R
Rank_2Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_2R
Rank_3Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_3X
sub_1SubRank_2:output:0Rank_3:output:0*
T0*
_output_shapes
: 2
sub_1M
Shape_3ShapeEqual_1:z:0*
T0
*
_output_shapes
:2	
Shape_3
ones_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2
ones_1/Reshape/shapez
ones_1/ReshapeReshape	sub_1:z:0ones_1/Reshape/shape:output:0*
T0*
_output_shapes
:2
ones_1/Reshape^
ones_1/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ones_1/Constm
ones_1Fillones_1/Reshape:output:0ones_1/Const:output:0*
T0*
_output_shapes
:2
ones_1`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_5/axis
concat_5ConcatV2Shape_3:output:0ones_1:output:0concat_5/axis:output:0*
N*
T0*
_output_shapes
:2

concat_5s
	Reshape_1ReshapeEqual_1:z:0concat_5:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2
	Reshape_1

SelectV2_2SelectV2Reshape_1:output:0zeros_2:output:0SelectV2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_2

SelectV2_3SelectV2Reshape_1:output:0zeros_3:output:0SelectV2_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

SelectV2_3z
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims/dimЊ
QRnnNetwork/ExpandDims
ExpandDimsobservation_pos#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims~
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_1/dimЖ
QRnnNetwork/ExpandDims_1
ExpandDimsobservation_price%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*/
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_1~
QRnnNetwork/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B :2
QRnnNetwork/ExpandDims_2/dimІ
QRnnNetwork/ExpandDims_2
ExpandDims	step_type%QRnnNetwork/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/ExpandDims_2С
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	21
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeУ
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   29
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ23
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeЧ
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeShape!QRnnNetwork/ExpandDims_1:output:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_flatten_1/ShapeЫ
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"џџџџ      2;
9QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape
3QRnnNetwork/EncodingNetwork/batch_flatten_1/ReshapeReshape!QRnnNetwork/ExpandDims_1:output:0BQRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ25
3QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshapeе
&QRnnNetwork/EncodingNetwork/dense/CastCast:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/EncodingNetwork/dense/Castѓ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype029
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp§
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul*QRnnNetwork/EncodingNetwork/dense/Cast:y:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/EncodingNetwork/dense/MatMulђ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/EncodingNetwork/dense/BiasAddЇ
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџd   2+
)QRnnNetwork/EncodingNetwork/flatten/Const
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape<QRnnNetwork/EncodingNetwork/batch_flatten_1/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџd2-
+QRnnNetwork/EncodingNetwork/flatten/ReshapeЌ
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :25
3QRnnNetwork/EncodingNetwork/concatenate/concat/axisЯ
.QRnnNetwork/EncodingNetwork/concatenate/concatConcatV22QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:04QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0<QRnnNetwork/EncodingNetwork/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџf20
.QRnnNetwork/EncodingNetwork/concatenate/concatЋ
+QRnnNetwork/EncodingNetwork/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџf   2-
+QRnnNetwork/EncodingNetwork/flatten_1/Const
-QRnnNetwork/EncodingNetwork/flatten_1/ReshapeReshape7QRnnNetwork/EncodingNetwork/concatenate/concat:output:04QRnnNetwork/EncodingNetwork/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf2/
-QRnnNetwork/EncodingNetwork/flatten_1/Reshapeљ
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:f(*
dtype02;
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul6QRnnNetwork/EncodingNetwork/flatten_1/Reshape:output:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2,
*QRnnNetwork/EncodingNetwork/dense_1/MatMulј
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:(*
dtype02<
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ(2-
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddФ
(QRnnNetwork/EncodingNetwork/dense_1/ReluRelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ(2*
(QRnnNetwork/EncodingNetwork/dense_1/ReluЬ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1а
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2ш
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice:QRnnNetwork/EncodingNetwork/batch_flatten_1/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2;
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceм
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	23
1QRnnNetwork/EncodingNetwork/batch_unflatten/Shapeа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2C
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackд
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1д
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2№
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_mask2=
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1Д
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisю
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:24
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatЎ
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Relu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
T0*
Tshape0	*+
_output_shapes
:џџџџџџџџџ(25
3QRnnNetwork/EncodingNetwork/batch_unflatten/Reshapej
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 2
QRnnNetwork/mask/y
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_2:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/mask
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :2!
QRnnNetwork/dynamic_unroll/Rank
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/start
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/range/deltaѕ
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/rangeЉ
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       2,
*QRnnNetwork/dynamic_unroll/concat/values_0
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&QRnnNetwork/dynamic_unroll/concat/axis
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!QRnnNetwork/dynamic_unroll/concatљ
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ(2&
$QRnnNetwork/dynamic_unroll/transpose
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
:2"
 QRnnNetwork/dynamic_unroll/ShapeЊ
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:20
.QRnnNetwork/dynamic_unroll/strided_slice/stackЎ
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Ў
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(QRnnNetwork/dynamic_unroll/strided_sliceЋ
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2-
+QRnnNetwork/dynamic_unroll/transpose_1/permл
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџ2(
&QRnnNetwork/dynamic_unroll/transpose_1
&QRnnNetwork/dynamic_unroll/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2(
&QRnnNetwork/dynamic_unroll/zeros/mul/yи
$QRnnNetwork/dynamic_unroll/zeros/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:0/QRnnNetwork/dynamic_unroll/zeros/mul/y:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dynamic_unroll/zeros/mul
'QRnnNetwork/dynamic_unroll/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2)
'QRnnNetwork/dynamic_unroll/zeros/Less/yг
%QRnnNetwork/dynamic_unroll/zeros/LessLess(QRnnNetwork/dynamic_unroll/zeros/mul:z:00QRnnNetwork/dynamic_unroll/zeros/Less/y:output:0*
T0*
_output_shapes
: 2'
%QRnnNetwork/dynamic_unroll/zeros/Less
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/zeros/packed/1я
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2)
'QRnnNetwork/dynamic_unroll/zeros/packed
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&QRnnNetwork/dynamic_unroll/zeros/Constс
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2"
 QRnnNetwork/dynamic_unroll/zeros
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2*
(QRnnNetwork/dynamic_unroll/zeros_1/mul/yо
&QRnnNetwork/dynamic_unroll/zeros_1/mulMul1QRnnNetwork/dynamic_unroll/strided_slice:output:01QRnnNetwork/dynamic_unroll/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2(
&QRnnNetwork/dynamic_unroll/zeros_1/mul
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2+
)QRnnNetwork/dynamic_unroll/zeros_1/Less/yл
'QRnnNetwork/dynamic_unroll/zeros_1/LessLess*QRnnNetwork/dynamic_unroll/zeros_1/mul:z:02QRnnNetwork/dynamic_unroll/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2)
'QRnnNetwork/dynamic_unroll/zeros_1/Less
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1ѕ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2+
)QRnnNetwork/dynamic_unroll/zeros_1/packed
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(QRnnNetwork/dynamic_unroll/zeros_1/Constщ
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"QRnnNetwork/dynamic_unroll/zeros_1Ц
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*'
_output_shapes
:џџџџџџџџџ(*
squeeze_dims
 2$
"QRnnNetwork/dynamic_unroll/SqueezeШ
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 2&
$QRnnNetwork/dynamic_unroll/Squeeze_1ё
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0SelectV2_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2#
!QRnnNetwork/dynamic_unroll/Selectї
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0SelectV2_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2%
#QRnnNetwork/dynamic_unroll/Select_1ќ
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource*
_output_shapes

:(P*
dtype02<
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2-
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMul
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

:P*
dtype02>
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2/
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1џ
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:џџџџџџџџџP2*
(QRnnNetwork/dynamic_unroll/lstm_cell/addћ
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02=
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP2.
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2,
*QRnnNetwork/dynamic_unroll/lstm_cell/ConstЎ
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :26
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimг
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*
	num_split2,
*QRnnNetwork/dynamic_unroll/lstm_cell/splitЮ
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoidв
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1я
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2*
(QRnnNetwork/dynamic_unroll/lstm_cell/mulХ
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџ2+
)QRnnNetwork/dynamic_unroll/lstm_cell/Tanhђ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1ё
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1в
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Ф
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2-
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1і
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџ2,
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)QRnnNetwork/dynamic_unroll/ExpandDims/dimі
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2'
%QRnnNetwork/dynamic_unroll/ExpandDimsв
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02.
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2$
"QRnnNetwork/dense_2/Tensordot/axes
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2$
"QRnnNetwork/dense_2/Tensordot/freeЈ
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/Shape
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisЕ
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/GatherV2 
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisЛ
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2*
(QRnnNetwork/dense_2/Tensordot/GatherV2_1
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2%
#QRnnNetwork/dense_2/Tensordot/Constа
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2$
"QRnnNetwork/dense_2/Tensordot/Prod
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%QRnnNetwork/dense_2/Tensordot/Const_1и
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2&
$QRnnNetwork/dense_2/Tensordot/Prod_1
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)QRnnNetwork/dense_2/Tensordot/concat/axis
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2&
$QRnnNetwork/dense_2/Tensordot/concatм
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2%
#QRnnNetwork/dense_2/Tensordot/stackє
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2)
'QRnnNetwork/dense_2/Tensordot/transposeя
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2'
%QRnnNetwork/dense_2/Tensordot/Reshapeю
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2&
$QRnnNetwork/dense_2/Tensordot/MatMul
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%QRnnNetwork/dense_2/Tensordot/Const_2
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2-
+QRnnNetwork/dense_2/Tensordot/concat_1/axisЁ
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2(
&QRnnNetwork/dense_2/Tensordot/concat_1р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/TensordotШ
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpз
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/BiasAdd
QRnnNetwork/dense_2/ReluRelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2
QRnnNetwork/dense_2/Relu
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02?
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpД
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3QRnnNetwork/num_action_project/dense/Tensordot/axesЛ
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3QRnnNetwork/num_action_project/dense/Tensordot/freeТ
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_2/Relu:activations:0*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeО
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2Т
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1Ж
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4QRnnNetwork/num_action_project/dense/Tensordot/Const
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3QRnnNetwork/num_action_project/dense/Tensordot/ProdК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1К
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisщ
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5QRnnNetwork/num_action_project/dense/Tensordot/concat 
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4QRnnNetwork/num_action_project/dense/Tensordot/stack
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Relu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ2:
8QRnnNetwork/num_action_project/dense/Tensordot/transposeГ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeВ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ27
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulК
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:28
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2О
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisі
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1Є
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ20
.QRnnNetwork/num_action_project/dense/Tensordotћ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02=
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ2.
,QRnnNetwork/num_action_project/dense/BiasAddЕ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
2
QRnnNetwork/Squeeze
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2%
#Categorical_1/mode/ArgMax/dimensionК
Categorical_1/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/ArgMax
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџ2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xД
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2Щ
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsЯ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisЊ
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concatЮ
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2$
"Deterministic_1/sample/BroadcastTo
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ђ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackІ
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1І
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2ъ
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1а
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/yВ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0*#
_output_shapes
:џџџџџџџџџ2

Identity

Identity_1Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1

Identity_2Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*О
_input_shapesЌ
Љ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ::::::::::::N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameobservation/pos:^Z
+
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameobservation/price:JF
'
_output_shapes
:џџџџџџџџџ

_user_specified_name0:JF
'
_output_shapes
:џџџџџџџџџ

_user_specified_name1

\
,__inference_function_with_signature_17166931

batch_size
identity

identity_1М
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_get_initial_state_171669262
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityp

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
ч'

#__inference__traced_save_1033060397
file_prefix'
#savev2_variable_read_readvariableop	G
Csavev2_qrnnnetwork_encodingnetwork_dense_kernel_read_readvariableopE
Asavev2_qrnnnetwork_encodingnetwork_dense_bias_read_readvariableopI
Esavev2_qrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableopG
Csavev2_qrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableop@
<savev2_qrnnnetwork_dynamic_unroll_kernel_read_readvariableopJ
Fsavev2_qrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableop>
:savev2_qrnnnetwork_dynamic_unroll_bias_read_readvariableop9
5savev2_qrnnnetwork_dense_2_kernel_read_readvariableop7
3savev2_qrnnnetwork_dense_2_bias_read_readvariableopJ
Fsavev2_qrnnnetwork_num_action_project_dense_kernel_read_readvariableopH
Dsavev2_qrnnnetwork_num_action_project_dense_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_565b12e25461462aac8fae585da77da9/part2	
Const_1
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameУ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*е
valueЫBШB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЛ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopCsavev2_qrnnnetwork_encodingnetwork_dense_kernel_read_readvariableopAsavev2_qrnnnetwork_encodingnetwork_dense_bias_read_readvariableopEsavev2_qrnnnetwork_encodingnetwork_dense_1_kernel_read_readvariableopCsavev2_qrnnnetwork_encodingnetwork_dense_1_bias_read_readvariableop<savev2_qrnnnetwork_dynamic_unroll_kernel_read_readvariableopFsavev2_qrnnnetwork_dynamic_unroll_recurrent_kernel_read_readvariableop:savev2_qrnnnetwork_dynamic_unroll_bias_read_readvariableop5savev2_qrnnnetwork_dense_2_kernel_read_readvariableop3savev2_qrnnnetwork_dense_2_bias_read_readvariableopFsavev2_qrnnnetwork_num_action_project_dense_kernel_read_readvariableopDsavev2_qrnnnetwork_num_action_project_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*s
_input_shapesb
`: : :::f(:(:(P:P:P::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:f(: 

_output_shapes
:(:$ 

_output_shapes

:(P:$ 

_output_shapes

:P: 

_output_shapes
:P:$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: "ИL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*щ
actionо
4

0/discount&
action_0/discount:0џџџџџџџџџ
F
0/observation/pos1
action_0/observation/pos:0џџџџџџџџџ
N
0/observation/price7
action_0/observation/price:0џџџџџџџџџ
0
0/reward$
action_0/reward:0џџџџџџџџџ
6
0/step_type'
action_0/step_type:0џџџџџџџџџ
*
1/0#
action_1/0:0џџџџџџџџџ
*
1/1#
action_1/1:0џџџџџџџџџ6
action,
StatefulPartitionedCall:0џџџџџџџџџ;
state/00
StatefulPartitionedCall:1џџџџџџџџџ;
state/10
StatefulPartitionedCall:2џџџџџџџџџtensorflow/serving/predict*Ф
get_initial_stateЎ
2

batch_size$
get_initial_state_batch_size:0 -
0(
PartitionedCall:0џџџџџџџџџ-
1(
PartitionedCall:1џџџџџџџџџtensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:шг

collect_data_spec
policy_state_spec

train_step
metadata
model_variables
_all_assets

signatures
action
distribution
get_initial_state
get_metadata
get_train_step"
_generic_user_object
9
observation
1"
trackable_tuple_wrapper
 "
trackable_list_wrapper
:	 (2Variable
 "
trackable_dict_wrapper
o
	0

1
2
3
4
5
6
7
8
9
10"
trackable_tuple_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
d
action
get_initial_state
get_train_step
get_metadata"
signature_map
 "
trackable_dict_wrapper
::82(QRnnNetwork/EncodingNetwork/dense/kernel
4:22&QRnnNetwork/EncodingNetwork/dense/bias
<::f(2*QRnnNetwork/EncodingNetwork/dense_1/kernel
6:4(2(QRnnNetwork/EncodingNetwork/dense_1/bias
3:1(P2!QRnnNetwork/dynamic_unroll/kernel
=:;P2+QRnnNetwork/dynamic_unroll/recurrent_kernel
-:+P2QRnnNetwork/dynamic_unroll/bias
,:*2QRnnNetwork/dense_2/kernel
&:$2QRnnNetwork/dense_2/bias
=:;2+QRnnNetwork/num_action_project/dense/kernel
7:52)QRnnNetwork/num_action_project/dense/bias
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
9
observation
3"
trackable_tuple_wrapper
 "
trackable_list_wrapper
3
	state
1"
trackable_tuple_wrapper
9
observation
1"
trackable_tuple_wrapper


_q_network
_time_step_spec
_policy_state_spec
_policy_step_spec
 _trajectory_spec"
_generic_user_object

!_input_tensor_spec
_state_spec
"_input_encoder
#_lstm_network
$_output_encoder
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerў{"class_name": "QRnnNetwork", "name": "QRnnNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
3
	state
1"
trackable_tuple_wrapper
9
observation
1"
trackable_tuple_wrapper
 "
trackable_dict_wrapper
Л
)_input_tensor_spec
*_preprocessing_nest
+_flat_preprocessing_layers
,_preprocessing_combiner
-_postprocessing_layers
.regularization_losses
/	variables
0trainable_variables
1	keras_api
+&call_and_return_all_conditional_losses
 __call__" 
_tf_keras_layer{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
ј	
2cell
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+Ё&call_and_return_all_conditional_losses
Ђ__call__"н
_tf_keras_layerУ{"class_name": "DynamicUnroll", "name": "dynamic_unroll", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dynamic_unroll", "trainable": true, "dtype": "float32", "parallel_iterations": 20, "swap_memory": null, "cell": {"class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 40]}}
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
n
	0

1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
n
	0

1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
А
%regularization_losses
9layer_metrics
:non_trainable_variables

;layers
&	variables
'trainable_variables
<layer_regularization_losses
=metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
>0
?1"
trackable_list_wrapper
Х
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+Ѓ&call_and_return_all_conditional_losses
Є__call__"Д
_tf_keras_layer{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [1, 2]}, {"class_name": "TensorShape", "items": [1, 100]}]}
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
А
.regularization_losses
Flayer_metrics
Gnon_trainable_variables

Hlayers
/	variables
0trainable_variables
Ilayer_regularization_losses
Jmetrics
 __call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
І

kernel
recurrent_kernel
bias
Kregularization_losses
L	variables
Mtrainable_variables
N	keras_api
+Ѕ&call_and_return_all_conditional_losses
І__call__"щ
_tf_keras_layerЯ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
А
3regularization_losses
Olayer_metrics
Pnon_trainable_variables

Qlayers
4	variables
5trainable_variables
Rlayer_regularization_losses
Smetrics
Ђ__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
Ь

kernel
bias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+Ї&call_and_return_all_conditional_losses
Ј__call__"Ѕ
_tf_keras_layer{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 20]}}
ю

kernel
bias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
+Љ&call_and_return_all_conditional_losses
Њ__call__"Ч
_tf_keras_layer­{"class_name": "Dense", "name": "num_action_project/dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "num_action_project/dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 20]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
"0
#1
72
83"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ъ

	kernel

bias
\regularization_losses
]	variables
^trainable_variables
_	keras_api
+Ћ&call_and_return_all_conditional_losses
Ќ__call__"У
_tf_keras_layerЉ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
ф
`regularization_losses
a	variables
btrainable_variables
c	keras_api
+­&call_and_return_all_conditional_losses
Ў__call__"г
_tf_keras_layerЙ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
@regularization_losses
dlayer_metrics
enon_trainable_variables

flayers
A	variables
Btrainable_variables
glayer_regularization_losses
hmetrics
Є__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
ш
iregularization_losses
j	variables
ktrainable_variables
l	keras_api
+Џ&call_and_return_all_conditional_losses
А__call__"з
_tf_keras_layerН{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ы

kernel
bias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
+Б&call_and_return_all_conditional_losses
В__call__"Є
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 40, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 102}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 102]}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
>0
?1
,2
D3
E4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
А
Kregularization_losses
qlayer_metrics
rnon_trainable_variables

slayers
L	variables
Mtrainable_variables
tlayer_regularization_losses
umetrics
І__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
20"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
Tregularization_losses
vlayer_metrics
wnon_trainable_variables

xlayers
U	variables
Vtrainable_variables
ylayer_regularization_losses
zmetrics
Ј__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
А
Xregularization_losses
{layer_metrics
|non_trainable_variables

}layers
Y	variables
Ztrainable_variables
~layer_regularization_losses
metrics
Њ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
Е
\regularization_losses
layer_metrics
non_trainable_variables
layers
]	variables
^trainable_variables
 layer_regularization_losses
metrics
Ќ__call__
+Ћ&call_and_return_all_conditional_losses
'Ћ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
`regularization_losses
layer_metrics
non_trainable_variables
layers
a	variables
btrainable_variables
 layer_regularization_losses
metrics
Ў__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
iregularization_losses
layer_metrics
non_trainable_variables
layers
j	variables
ktrainable_variables
 layer_regularization_losses
metrics
А__call__
+Џ&call_and_return_all_conditional_losses
'Џ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Е
mregularization_losses
layer_metrics
non_trainable_variables
layers
n	variables
otrainable_variables
 layer_regularization_losses
metrics
В__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
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
2
__inference_action_1259
__inference_action_17167228Ч
ОВК
FullArgSpec8
args0-
jself
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaultsЂ	
Ђ 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
 __inference_distribution_fn_1505Ћ
ЄВ 
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
!__inference_get_initial_state_984І
В
FullArgSpec!
args
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
__inference_<lambda>_993
B
__inference_<lambda>_990
B
(__inference_signature_wrapper_1033060305
0/discount0/observation/pos0/observation/price0/reward0/step_type1/01/1
:B8
(__inference_signature_wrapper_1033060314
batch_size
,B*
(__inference_signature_wrapper_1033060322
,B*
(__inference_signature_wrapper_1033060326
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
т2пм
гВЯ
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults	
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2ур
зВг
FullArgSpecL
argsDA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults

 
Ђ 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
FullArgSpecH
args@=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
с2ол
вВЮ
FullArgSpecH
args@=
jself
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaults

 

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ф2СО
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 7
__inference_<lambda>_990Ђ

Ђ 
Њ " 	0
__inference_<lambda>_993Ђ

Ђ 
Њ "Њ Ў
__inference_action_1259	
цЂт
кЂж
В
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ~
observationoЊl
0
pos)&
observation/posџџџџџџџџџ
8
price/,
observation/priceџџџџџџџџџ
=:

0џџџџџџџџџ

1џџџџџџџџџ

 
Њ "В

PolicyStep&
action
actionџџџџџџџџџR
stateIF
!
state/0џџџџџџџџџ
!
state/1џџџџџџџџџ
infoЂ 
__inference_action_17167228с	
ЕЂБ
ЉЂЅ
ХВС
TimeStep6
	step_type)&
time_step/step_typeџџџџџџџџџ0
reward&#
time_step/rewardџџџџџџџџџ4
discount(%
time_step/discountџџџџџџџџџ
observationЊ
:
pos30
time_step/observation/posџџџџџџџџџ
B
price96
time_step/observation/priceџџџџџџџџџ
WT
(%
policy_state/0џџџџџџџџџ
(%
policy_state/1џџџџџџџџџ

 
Њ "В

PolicyStep&
action
actionџџџџџџџџџR
stateIF
!
state/0џџџџџџџџџ
!
state/1џџџџџџџџџ
infoЂ 
 __inference_distribution_fn_1505ѓ	
тЂо
жЂв
В
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ~
observationoЊl
0
pos)&
observation/posџџџџџџџџџ
8
price/,
observation/priceџџџџџџџџџ
=:

0џџџџџџџџџ

1џџџџџџџџџ
Њ "ўВњ

PolicyStep
actionџћ№сУлЂз
`
CЂ@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*Њ'
%
loc
Identityџџџџџџџџџ
`Њ]

allow_nan_statsp


atol
 

namejDeterministic


rtol
 

validate_argsp _DistributionTypeSpecR
stateIF
!
state/0џџџџџџџџџ
!
state/1џџџџџџџџџ
infoЂ 
!__inference_get_initial_state_984c"Ђ
Ђ


batch_size 
Њ "=:

0џџџџџџџџџ

1џџџџџџџџџО
(__inference_signature_wrapper_1033060305	
іЂђ
Ђ 
ъЊц
.

0/discount 

0/discountџџџџџџџџџ
@
0/observation/pos+(
0/observation/posџџџџџџџџџ
H
0/observation/price1.
0/observation/priceџџџџџџџџџ
*
0/reward
0/rewardџџџџџџџџџ
0
0/step_type!
0/step_typeџџџџџџџџџ
$
1/0
1/0џџџџџџџџџ
$
1/1
1/1џџџџџџџџџ"Њ
&
action
actionџџџџџџџџџ
,
state/0!
state/0џџџџџџџџџ
,
state/1!
state/1џџџџџџџџџЇ
(__inference_signature_wrapper_1033060314{0Ђ-
Ђ 
&Њ#
!

batch_size

batch_size "GЊD
 
0
0џџџџџџџџџ
 
1
1џџџџџџџџџ\
(__inference_signature_wrapper_10330603220Ђ

Ђ 
Њ "Њ

int64
int64 	@
(__inference_signature_wrapper_1033060326Ђ

Ђ 
Њ "Њ 