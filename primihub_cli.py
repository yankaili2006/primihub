#!/usr/bin/env python3
"""
PrimiHub CLI — 全功能命令行工具
覆盖: PSI/PIR/FL/MPC/联邦查询/统计/特征工程/数据变换
"""
import sys, json, os, argparse, inspect, importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))
from primihub import context as ctx
from primihub import pyclient

# ─── PSI / PSU ───
def cmd_psi(args):
    """PSI 隐私求交"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.MPC.psi import run_psi
    result = run_psi(c, role=args.role, party=args.party, dataset=args.dataset, protocol=args.protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_psu(args):
    """PSU 隐私求并"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.psu import run_psu
    result = run_psu(c, role=args.role, party=args.party, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Federated Query ───
def cmd_query(args):
    """联邦查询"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.federated_query import run_query
    result = run_query(c, sql=args.sql, dataset=args.dataset, protocol=args.protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Federated Learning ───
def cmd_fl_lr(args):
    """联邦线性回归"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.linear_regression import run_lr
    result = run_lr(c, role=args.role, dataset=args.dataset, algorithm=args.algorithm)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_fl_logistic(args):
    """联邦逻辑回归"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.logistic_regression import run_logistic
    result = run_logistic(c, role=args.role, dataset=args.dataset, algorithm=args.algorithm)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_fl_xgb(args):
    """联邦 XGBoost"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.xgboost import run_xgb
    result = run_xgb(c, role=args.role, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_fl_nn(args):
    """联邦神经网络"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.neural_network import run_nn
    result = run_nn(c, role=args.role, dataset=args.dataset, model_type=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Feature Engineering ───
def cmd_feature_similarity(args):
    """特征相似度分析"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.feature_similarity import run
    result = run(c, role=args.role, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_feature_encoding(args):
    """特征编码"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.feature_encoding import run_encoding
    result = run_encoding(c, dataset=args.dataset, method=args.method)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_feature_binning(args):
    """特征分箱"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.feature_binning import run_binning
    result = run_binning(c, dataset=args.dataset, strategy=args.strategy, bins=args.bins)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_feature_selection(args):
    """特征筛选"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL import feature_selection
    result = feature_selection.run(c, dataset=args.dataset, method=args.method)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_feature_derivation(args):
    """特征衍生"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL import feature_derivation
    result = feature_derivation.run(c, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_feature_imputation(args):
    """特征填充（缺失值处理）"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.feature_imputation import run_imputation
    result = run_imputation(c, dataset=args.dataset, strategy=args.strategy)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_feature_alignment(args):
    """特征对齐"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.feature_alignment import run_alignment
    result = run_alignment(c, role=args.role, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_feature_sharing(args):
    """特征分享"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.feature_sharing import run_sharing
    result = run_sharing(c, role=args.role, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Data Transformation ───
def cmd_data_split(args):
    """数据分割"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.data_splitting import run_split
    result = run_split(c, dataset=args.dataset, ratio=args.ratio)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_data_transform(args):
    """数据转换"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.data_transformation import run_transform
    result = run_transform(c, dataset=args.dataset, method=args.method)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_data_fusion(args):
    """数据融合"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.data_fusion import run_fusion
    result = run_fusion(c, role=args.role, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── MPC ───
def cmd_mpc_stats(args):
    """MPC 安全统计"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.MPC.statistics import run_statistics
    result = run_statistics(c, dataset=args.dataset, stat_type=args.type)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_mpc_express(args):
    """MPC 表达式计算"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.MPC.express import run_express
    result = run_express(c, expression=args.expression, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Preprocessing ───
def cmd_preprocess(args):
    """联邦预处理"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.fl_preprocessing import run_preprocessing
    result = run_preprocessing(c, dataset=args.dataset, ops=args.operations)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_sample_expand(args):
    """样本扩展"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.sample_expansion import run_expansion
    result = run_expansion(c, dataset=args.dataset, method=args.method)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_sample_weight(args):
    """样本加权"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.sample_weighting import run_weighting
    result = run_weighting(c, dataset=args.dataset, weight=args.weight)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Metrics ───
def cmd_metrics(args):
    """模型评估指标"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL import metrics
    result = metrics.evaluate(c, pred=args.prediction, truth=args.truth, metric=args.metric)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_metrics_modeling(args):
    """指标建模分析"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.metrics_modeling import run_metrics
    result = run_metrics(c, dataset=args.dataset, target=args.target)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Model Management ───
def cmd_model_eval(args):
    """模型评估"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.model_evaluation import run_evaluation
    result = run_evaluation(c, model=args.model, dataset=args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_sketch(args):
    """数据草图/概要统计"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.sketch import run_sketch
    result = run_sketch(c, dataset=args.dataset, sketch_type=args.type)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ─── Crypto ───
def cmd_crypto(args):
    """密码学操作（同态加密密钥生成/加密/解密）"""
    c = ctx.Context(node=args.node, port=args.port)
    from primihub.FL.crypto import run_crypto
    result = run_crypto(c, operation=args.op, data=args.data, key=args.key)
    print(json.dumps(result, ensure_ascii=False, indent=2))

# ═══ Main CLI ═══
def main():
    parser = argparse.ArgumentParser(description="PrimiHub CLI — 全功能隐私计算命令行工具")
    parser.add_argument("--node", default="localhost", help="节点地址")
    parser.add_argument("--port", default=50050, type=int, help="节点端口")
    parser.add_argument("--role", default=0, type=int, help="角色 (0=host, 1=guest)")
    parser.add_argument("--party", help="参与方配置")
    parser.add_argument("--dataset", help="数据集路径")
    parser.add_argument("--protocol", default="ECDH", help="协议 (ECDH/KKRT/OT)")
    parser.add_argument("--algorithm", default="plaintext", help="算法类型")
    parser.add_argument("--method", help="方法名")
    parser.add_argument("--strategy", default="mean", help="策略")
    parser.add_argument("--bins", type=int, default=10, help="分箱数")
    parser.add_argument("--ratio", type=float, default=0.8, help="分割比例")
    parser.add_argument("--model", help="模型类型/路径")
    parser.add_argument("--sql", help="SQL 语句")
    parser.add_argument("--expression", help="MPC 表达式")
    parser.add_argument("--operations", help="预处理操作列表")
    parser.add_argument("--weight", type=float, default=1.0, help="权重")
    parser.add_argument("--prediction", help="预测值路径")
    parser.add_argument("--truth", help="真实值路径")
    parser.add_argument("--metric", default="auc", help="评估指标")
    parser.add_argument("--target", help="目标列")
    parser.add_argument("--type", help="类型 (统计/草图)")
    parser.add_argument("--op", help="密码学操作 (generate/encrypt/decrypt)")
    parser.add_argument("--data", help="数据")
    parser.add_argument("--key", help="密钥")

    sub = parser.add_subparsers(dest="command", title="命令")

    # PSI / PSU
    p = sub.add_parser("psi", help="隐私求交 (Private Set Intersection)")
    p.add_argument("--protocol", default="ECDH", choices=["ECDH", "KKRT", "OT"])
    p.set_defaults(func=cmd_psi)

    p = sub.add_parser("psu", help="隐私求并 (Private Set Union)")
    p.set_defaults(func=cmd_psu)

    p = sub.add_parser("query", help="联邦查询")
    p.add_argument("--sql", required=True)
    p.add_argument("--protocol", default="DH", choices=["DH", "OT", "HE"])
    p.set_defaults(func=cmd_query)

    # Federated Learning
    p = sub.add_parser("fl-lr", help="联邦线性回归")
    p.add_argument("--algorithm", default="plaintext", choices=["plaintext", "Paillier", "CKKS"])
    p.set_defaults(func=cmd_fl_lr)

    p = sub.add_parser("fl-logistic", help="联邦逻辑回归")
    p.add_argument("--algorithm", default="plaintext", choices=["plaintext", "Paillier", "DPSGD"])
    p.set_defaults(func=cmd_fl_logistic)

    p = sub.add_parser("fl-xgb", help="联邦 XGBoost")
    p.set_defaults(func=cmd_fl_xgb)

    p = sub.add_parser("fl-nn", help="联邦神经网络")
    p.add_argument("--model", default="classifier", choices=["classifier", "regressor"])
    p.set_defaults(func=cmd_fl_nn)

    # Feature Engineering
    p = sub.add_parser("feature-similarity", help="特征相似度分析")
    p.set_defaults(func=cmd_feature_similarity)

    p = sub.add_parser("feature-encode", help="特征编码")
    p.add_argument("--method", default="onehot", choices=["onehot", "label", "woe"])
    p.set_defaults(func=cmd_feature_encoding)

    p = sub.add_parser("feature-bin", help="特征分箱")
    p.add_argument("--strategy", default="equal_freq", choices=["equal_freq", "equal_width", "kmeans"])
    p.add_argument("--bins", type=int, default=10)
    p.set_defaults(func=cmd_feature_binning)

    p = sub.add_parser("feature-select", help="特征筛选")
    p.add_argument("--method", default="iv", choices=["iv", "vif", "chi2", "mutual_info"])
    p.set_defaults(func=cmd_feature_selection)

    p = sub.add_parser("feature-derive", help="特征衍生")
    p.set_defaults(func=cmd_feature_derivation)

    p = sub.add_parser("feature-impute", help="缺失值填充")
    p.add_argument("--strategy", default="mean", choices=["mean", "median", "most_frequent", "constant"])
    p.set_defaults(func=cmd_feature_imputation)

    p = sub.add_parser("feature-align", help="特征对齐")
    p.set_defaults(func=cmd_feature_alignment)

    p = sub.add_parser("feature-share", help="特征分享")
    p.set_defaults(func=cmd_feature_sharing)

    # Data Transformation
    p = sub.add_parser("data-split", help="数据分割")
    p.add_argument("--ratio", type=float, default=0.8)
    p.set_defaults(func=cmd_data_split)

    p = sub.add_parser("data-transform", help="数据转换")
    p.add_argument("--method", default="standard", choices=["standard", "minmax", "robust", "log"])
    p.set_defaults(func=cmd_data_transform)

    p = sub.add_parser("data-fusion", help="数据融合")
    p.set_defaults(func=cmd_data_fusion)

    # MPC
    p = sub.add_parser("mpc-stats", help="MPC 安全统计")
    p.add_argument("--type", default="sum", choices=["sum", "avg", "max", "min", "var", "std"])
    p.set_defaults(func=cmd_mpc_stats)

    p = sub.add_parser("mpc-express", help="MPC 表达式计算")
    p.add_argument("--expression", required=True)
    p.set_defaults(func=cmd_mpc_express)

    # Preprocessing
    p = sub.add_parser("preprocess", help="联邦预处理")
    p.add_argument("--operations", default="normalize")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("sample-expand", help="样本扩展")
    p.add_argument("--method", default="smote", choices=["smote", "adasyn", "random"])
    p.set_defaults(func=cmd_sample_expand)

    p = sub.add_parser("sample-weight", help="样本加权")
    p.add_argument("--weight", type=float, default=1.0)
    p.set_defaults(func=cmd_sample_weight)

    # Metrics & Model
    p = sub.add_parser("evaluate", help="模型评估")
    p.add_argument("--prediction", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--metric", default="auc", choices=["auc", "acc", "f1", "precision", "recall", "mse", "mae", "r2"])
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("model-eval", help="模型评估（联邦）")
    p.add_argument("--model", required=True)
    p.set_defaults(func=cmd_model_eval)

    p = sub.add_parser("metrics-modeling", help="指标建模分析")
    p.add_argument("--target", required=True)
    p.set_defaults(func=cmd_metrics_modeling)

    p = sub.add_parser("sketch", help="数据概要统计")
    p.add_argument("--type", default="histogram", choices=["histogram", "quantile", "frequency", "topk"])
    p.set_defaults(func=cmd_sketch)

    # Crypto
    p = sub.add_parser("crypto", help="密码学操作")
    p.add_argument("--op", required=True, choices=["generate", "encrypt", "decrypt"])
    p.set_defaults(func=cmd_crypto)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
