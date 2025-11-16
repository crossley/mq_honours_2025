from imports import *
from util_func import *

dp = load_data()

# NOTE: inspect individual subjects --- measures
# for i, s in enumerate(dp["subject"].unique()):
# 
#     ds = dp[dp["subject"] == s].copy()
# 
#     fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(5, 12))
#     fig.subplots_adjust(wspace=0.3, hspace=0.5)
#
#     sns.scatterplot(
#         data=ds,
#         x="trial",
#         y="emv",
#         hue="su_prev",
#         markers=True,
#         legend="full",
#         ax=ax[0, 0],
#     )
#     sns.scatterplot(
#         data=ds,
#         x="trial",
#         y="movement_error",
#         hue="su_prev",
#         markers=True,
#         legend=False,
#         ax=ax[1, 0],
#     )
#     sns.scatterplot(
#         data=ds,
#         x="trial",
#         y="delta_emv",
#         hue="su_prev",
#         markers=True,
#         legend=False,
#         ax=ax[2, 0],
#     )
#     [
#         sns.lineplot(
#             data=ds,
#             x="trial",
#             y="rotation",
#             hue="condition",
#             palette=['k'],
#             legend=False,
#             ax=ax_,
#         ) for ax_ in [ax[0, 0], ax[1, 0], ax[2, 0]]
#     ]
#
#     ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
#
#     plt.savefig("../figures/fig_measures_sub_" + str(s) + ".png")
#     plt.close()

# # NOTE: inspect individual subjects --- scatter
# for i, s in enumerate(dp["subject"].unique()):
#
#     ds = dp[dp["subject"] == s].copy()
#
#     fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))
#     fig.subplots_adjust(wspace=0.3, hspace=0.5)
#
#     sns.scatterplot(
#         data=ds,
#         x="trial",
#         y="emv",
#         hue="su_prev",
#         markers=True,
#         legend="full",
#         ax=ax[0, 0],
#     )
#     sns.lineplot(data=ds,
#                  x="trial",
#                  y="rotation",
#                  hue="condition",
#                  palette=['k'],
#                  legend=False,
#                  ax=ax[0, 0])
#
#     for j, ph in enumerate([2, 5]):
#         ds = ds[~ds["su_prev"].isna()]
#         su_levels = np.sort(ds["su_prev"].unique())
#         dss = ds[ds["phase"] == ph].copy()
#         for k, su in enumerate(dss["su_prev"].unique()):
#             dsss = dss[dss["su_prev"] == su].copy()
#             sns.scatterplot(
#                 data=dsss,
#                 x="movement_error_prev",
#                 y="delta_emv",
#                 hue="su_prev",
#                 legend="full",
#                 ax=ax[0, j + 1],
#             )
#             sns.regplot(
#                 data=dsss,
#                 x="movement_error_prev",
#                 y="delta_emv",
#                 scatter=False,
#                 color=sns.color_palette()[np.where(su_levels == su)[0][0]],
#                 ax=ax[0, j + 1],
#             )
#
#     ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
#     ax[0, 1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
#     ax[0, 2].legend(loc="upper left", bbox_to_anchor=(0.0, 1.4), ncol=2)
#
#     plt.savefig("../figures/fig_scatter_sub_" + str(s) + ".png")
#     plt.close()

# NOTE: Exclude ppts that have abberant movements based on previous two figures
subs_exc = [5, 10, 30, 35, 40, 57]
dp = dp[~np.isin(dp["subject"], subs_exc)]

# NOTE: average over subjects
dpp = dp.groupby(["condition", "trial", "phase", "su_prev"], observed=True)[[
    "emv", "delta_emv", "movement_error", "movement_error_prev", "rotation"
]].mean().reset_index()

# dp.to_csv("../data_summary/summary_per_trial_per_subject.csv")
# dpp.to_csv("../data_summary/summary_per_trial.csv")

# fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(8, 12))
# ax = ax.T
# fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.95, bottom=0.05)
# sns.scatterplot(
#     data=dpp[dpp["condition"] == "Blocked - High Low"],
#     x="trial",
#     y="emv",
#     style="phase",
#     hue="su_prev",
#     markers=True,
#     legend="full",
#     ax=ax[0, 0],
# )
# sns.scatterplot(
#     data=dpp[dpp["condition"] == "Blocked - Low High"],
#     x="trial",
#     y="emv",
#     style="phase",
#     hue="su_prev",
#     markers=True,
#     legend="full",
#     ax=ax[0, 1],
# )
# sns.scatterplot(
#     data=dpp[dpp["condition"] == "interleaved"],
#     x="trial",
#     y="emv",
#     style="phase",
#     hue="su_prev",
#     markers=True,
#     legend="full",
#     ax=ax[0, 2],
# )
# [x.set_ylim(-10, 40) for x in [ax[0, 0], ax[0, 1], ax[0, 2]]]
# [x.set_xlabel("Trial") for x in [ax[0, 0], ax[0, 1], ax[0, 2]]]
# [
#     x.set_ylabel("Endppoint Movement Vector")
#     for x in [ax[0, 0], ax[0, 1], ax[0, 2]]
# ]
# [
#     sns.lineplot(
#         data=dpp[dpp["condition"] != "interleaved"],
#         x="trial",
#         y="rotation",
#         hue="condition",
#         palette=['k'],
#         legend=False,
#         ax=ax_,
#     ) for ax_ in [ax[0, 0], ax[0, 1], ax[0, 2]]
# ]
# ax[0, 0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
# ax[0, 1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
# ax[0, 2].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4)
# plt.savefig("../figures/summary_per_trial.png")
# plt.close()

# NOTE: scatter slope analysis

# adapt 1 is trials 30:130
ph = 2
dppp1 = dp[(dp["condition"] == "Blocked - Low High") & (dp["phase"] == ph) & (dp["trial"] < 35)].copy()
dppp2 = dp[(dp["condition"] == "Blocked - High Low") & (dp["phase"] == ph) & (dp["trial"] < 35)].copy()
dppp3 = dp[(dp["condition"] == "interleaved") & (dp["phase"] == ph) & (dp["trial"] < 35)].copy()

# fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))
# fig.subplots_adjust(wspace=0.3, hspace=0.5)
# sns.scatterplot(
#     data=dppp1,
#     x="movement_error_prev",
#     y="delta_emv",
#     hue="su_prev",
#     markers=True,
#     legend="full",
#     ax=ax[0, 0],
# )
# sns.scatterplot(
#     data=dppp2,
#     x="movement_error_prev",
#     y="delta_emv",
#     hue="su_prev",
#     markers=True,
#     legend="full",
#     ax=ax[0, 1],
# )
# sns.scatterplot(
#     data=dppp3,
#     x="movement_error_prev",
#     y="delta_emv",
#     hue="su_prev",
#     markers=True,
#     legend="full",
#     ax=ax[0, 2],
# )
# ax[0, 0].set_title("Blocked - Low High")
# ax[0, 1].set_title("Blocked - High Low")
# ax[0, 2].set_title("Interleaved")
# [x.set_xlim(-20, 5) for x in ax.flatten()]
# [x.set_ylim(-20, 20) for x in ax.flatten()]
# [x.set_xlabel("Movement error previous trial") for x in ax.flatten()]
# [x.set_ylabel("Change in EMV") for x in ax.flatten()]
# plt.savefig("../figures/summary_scatter_slope.png")
# plt.close()


# NOTE: statsmodels
def fit_regression(d):

    d["exp_fast"] = 1 - np.exp(-0.3 * d["trial"])
    d["exp_med"] = 1 - np.exp(-0.03 * d["trial"])
    md = smf.mixedlm(
        "emv ~ C(su_prev, Diff)*movement_error_prev + exp_fast + exp_med",
        data=d,
        groups=d["subject"])

    mdf = md.fit()
    print(mdf.summary())

    d["emv_pred"] = mdf.model.predict(mdf.params, mdf.model.exog)

    # plot obs and pred overliad
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(6, 6))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    sns.lineplot(data=d,
                 x="trial",
                 y="emv",
                 hue="su_prev",
                 markers=True,
                 ax=ax[0, 0])
    sns.lineplot(data=d,
                 x="trial",
                 y="emv_pred",
                 hue="su_prev",
                 markers=True,
                 ax=ax[0, 1])
    [x.set_ylim(-5, 20) for x in [ax[0, 0], ax[0, 1]]]
    plt.show()


# NOTE: set condition here
dpp = dp[(dp["condition"] != "interleaved") & (dp["phase"] == ph)].copy()
fit_regression(dpp)

dpp = dp[(dp["condition"] == "interleaved") & (dp["phase"] == ph)].copy()
fit_regression(dpp)


