import pandas as pd
from tqdm import tqdm


def create_pred_str_study(row):
    items = []
    for label in ["negative", "typical", "indeterminate", "atypical"]:
        items += [label, row[label], "0", "0", "1", "1"]
    return " ".join([str(i) for i in items])


def create_pred_str_image(det, min_conf, scale=(4, 4, 4, 4)):

    detection = det[det[:, 4] > min_conf]
    items = []
    for d in detection:
        items += ["opacity", d[4]] + list(d[:4] * scale)
    items += ["none", 1 - det[:, 4].max(), "0", "0", "1", "1"]
    return " ".join([str(i) for i in items])


def create_submission(cfg, results, dataset, pre="test"):

    test_df = dataset.df.copy()
    study_ids = test_df["StudyInstanceUID"].values

    if cfg.pp_transformation == "softmax":
        class_logits = results["class_logits"].softmax(1).detach().cpu().numpy()
    elif cfg.pp_transformation == "sigmoid":
        class_logits = results["class_logits"].sigmoid().detach().cpu().numpy()
    else:
        class_logits = results["class_logits"].detach().cpu().numpy()

    df_study = pd.DataFrame({"id": study_ids})
    df_study["id"] = df_study["id"].astype(str) + "_study"
    df_study[["negative", "typical", "indeterminate", "atypical"]] = class_logits
    df_study = df_study.groupby("id").agg("mean").reset_index()
    df_study["PredictionString"] = df_study.apply(lambda row: create_pred_str_study(row), axis=1)
    df_study = df_study[["id", "PredictionString"]].copy()

    test_df["x_scale"] = test_df["width"] / cfg.image_size[0]
    test_df["y_scale"] = test_df["height"] / cfg.image_size[1]

    detections = results["detections"].detach().cpu().numpy()
    prediction_strings = []
    for i, det in tqdm(enumerate(detections)):
        row = test_df.iloc[i]
        scale = (row["x_scale"], row["y_scale"], row["x_scale"], row["y_scale"])
        prediction_strings += [create_pred_str_image(det, cfg.box_min_conf, scale=scale)]

    test_df["PredictionString"] = prediction_strings
    df_image = test_df[["id", "PredictionString"]].copy()

    sub = pd.concat([df_image, df_study]).reset_index(drop=True)

    print("saving submission")
    sub.to_csv(f"{cfg.output_dir}/fold{cfg.fold}/submission_seed{cfg.seed}.csv", index=False)
