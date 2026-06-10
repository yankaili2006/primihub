package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "pir_inspect",
    srcs = ["pir_inspect_main.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:selector",
        # :pir is the aggregator (task 1.9) — pulls in registry +
        # default-enabled algorithm operators (id_pir, simple_pir,
        # double_pir, spiral_pir, frodo_pir, ypir; APSI opt-in via
        # //:microsoft-apsi select). Adding a new algorithm only
        # requires editing operator/BUILD :pir target.
        "//src/primihub/kernel/pir/operator:pir",
        # pir_inspect cache subcommand reads on-disk PHHC (DoublePIR)
        # / PSHC (SimplePIR) hint files via the serializer helpers;
        # these are utility libs, not algorithm operators.
        "//src/primihub/kernel/pir/operator/double_pir:hint_serialize",
        "//src/primihub/kernel/pir/operator/simple_pir:hint_serialize",
        "@com_github_glog_glog//:glog",
    ],
)
