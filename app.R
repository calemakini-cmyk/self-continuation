# ============================================================================
# Task 6 — Fraud Detection & XAI Dashboard (Final — all features engaged)
# Author: MAKINI CALEB
# ============================================================================
suppressPackageStartupMessages({
  library(shiny);     library(bslib);     library(tidyverse)
  library(plotly);    library(DT);        library(pROC)
  library(yardstick); library(DALEX);     library(caret)
  library(scales)
})

## ---- 1. Load artifacts -----------------------------------------------------
required <- c("credit_clean.csv","final_model_gbm.rds",
              "test_set.rds","model_comparison.rds")
missing  <- required[!file.exists(required)]
if (length(missing) > 0)
  stop("Missing: ", paste(missing, collapse = ", "), " in ", getwd())

data_clean  <- read.csv("credit_clean.csv") |>
  mutate(is_fraud            = factor(is_fraud,            levels = c("legit","fraud")),
         foreign_transaction = factor(foreign_transaction, levels = c("no","yes")),
         location_mismatch   = factor(location_mismatch,   levels = c("no","yes")))
final_model <- readRDS("final_model_gbm.rds")
test_set    <- readRDS("test_set.rds")
comparison  <- readRDS("model_comparison.rds")

## ---- 2. Precompute ---------------------------------------------------------
pred_fun   <- function(m, newdata) predict(m, newdata = newdata,
                                           type = "prob")[, "fraud"]
test_probs <- pred_fun(final_model, test_set)
y_test     <- as.integer(test_set$is_fraud == "fraud")
x_test     <- test_set |> select(-is_fraud)

explainer  <- DALEX::explain(final_model, data = x_test, y = y_test,
                             predict_function = pred_fun,
                             label = "Gradient Boosting",
                             type = "classification", verbose = FALSE)

gbm_imp <- tryCatch(
  caret::varImp(final_model)$importance |>
    tibble::rownames_to_column("Feature") |> arrange(desc(Overall)),
  error = function(e) data.frame(Feature = character(0), Overall = numeric(0)))

PAL        <- c(fraud = "#B22222", legit = "#4682B4")
CAT_LEVELS <- levels(x_test$merchant_category)
COST_FN    <- 500
COST_FP    <- 25

## ---- 3. Helpers ------------------------------------------------------------
prep_and_score <- function(df) {
  if ("transaction_id" %in% names(df)) df$transaction_id <- NULL
  actual <- NULL
  if ("is_fraud" %in% names(df)) {
    raw <- df$is_fraud
    actual <- if (is.numeric(raw))
      factor(ifelse(raw == 1, "fraud", "legit"), levels = c("legit","fraud"))
    else factor(as.character(raw), levels = c("legit","fraud"))
    df$is_fraud <- NULL
  }
  cb <- function(v) if (is.numeric(v)) factor(ifelse(v == 1, "yes", "no"),
                                              levels = c("no","yes"))
  else factor(as.character(v), levels = c("no","yes"))
  if ("foreign_transaction" %in% names(df)) df$foreign_transaction <- cb(df$foreign_transaction)
  if ("location_mismatch"   %in% names(df)) df$location_mismatch   <- cb(df$location_mismatch)
  if ("merchant_category"   %in% names(df))
    df$merchant_category <- factor(as.character(df$merchant_category), levels = CAT_LEVELS)
  df$impossible   <- df$amount <= 0
  df$micro_amount <- df$amount > 0 & df$amount < 0.10
  df$high_value   <- df$amount > 1200
  keep <- df$amount > 0 & !is.na(df$merchant_category)
  sc   <- df[keep, , drop = FALSE]
  n_dropped <- nrow(df) - nrow(sc)
  if (nrow(sc) > 0) {
    sc$p_fraud <- as.numeric(pred_fun(final_model, sc[, names(x_test), drop = FALSE]))
    sc$risk_band <- cut(sc$p_fraud, c(-Inf,.30,.55,Inf),
                        labels = c("LOW","MEDIUM","HIGH"))
    sc$recommended_action <- ifelse(sc$p_fraud < .30, "Auto-approve",
                                    ifelse(sc$p_fraud < .55, "Soft challenge / 2FA",
                                           "Block & investigate"))
    if (!is.null(actual)) sc$actual <- actual[keep]
  }
  list(df = sc, n_dropped = n_dropped, has_labels = !is.null(actual))
}

predict_one <- function(amount, hour, cat, foreign, mismatch,
                        trust, velocity, age) {
  nd <- tibble(
    amount = amount, transaction_hour = hour,
    merchant_category = factor(cat, levels = CAT_LEVELS),
    foreign_transaction = factor(foreign, levels = c("no","yes")),
    location_mismatch = factor(mismatch, levels = c("no","yes")),
    device_trust_score = trust, velocity_last_24h = velocity,
    cardholder_age = age,
    impossible = amount <= 0, micro_amount = amount > 0 & amount < 0.10,
    high_value = amount > 1200)
  list(prob = as.numeric(pred_fun(final_model, nd)), row = nd)
}

compute_local_drivers <- function(nd, reference) {
  base_p <- as.numeric(pred_fun(final_model, nd))
  feats  <- names(nd)
  deltas <- sapply(feats, function(col) {
    pert <- nd
    if (is.numeric(nd[[col]])) pert[[col]] <- median(reference[[col]], na.rm = TRUE)
    else {
      lvls <- levels(nd[[col]])
      if (!is.null(lvls)) {
        tab <- table(reference[[col]])
        pert[[col]] <- factor(names(tab)[which.max(tab)], levels = lvls)
      }
    }
    base_p - as.numeric(pred_fun(final_model, pert))
  })
  tibble(feature = feats,
         value = sapply(feats, function(col) as.character(nd[[col]][1])),
         delta = as.numeric(deltas)) |>
    mutate(direction = ifelse(delta > 0, "Increases risk", "Decreases risk"),
           pct = round(100 * delta, 2)) |> arrange(desc(abs(delta)))
}

demo_transaction <- function() {
  row <- test_set[sample(nrow(test_set), 1), ]
  list(amount = row$amount, hour = row$transaction_hour,
       cat = as.character(row$merchant_category),
       foreign = as.character(row$foreign_transaction),
       mismatch = as.character(row$location_mismatch),
       trust = row$device_trust_score, velocity = row$velocity_last_24h,
       age = row$cardholder_age, actual = as.character(row$is_fraud))
}

pct_rank <- function(x, v) round(100 * mean(x <= v, na.rm = TRUE), 1)
`%||%`   <- function(a, b) if (is.null(a) || length(a) == 0) b else a

## ---- 4. UI -----------------------------------------------------------------
ui <- page_navbar(
  id = "tabs",
  title = tags$span(icon("shield-halved"),
                    tags$b(" Fraud Shield"),
                    tags$small(class = "text-light ms-2", "v3.0")),
  theme = bs_theme(version = 5, bootswatch = "flatly", primary = "#4682B4"),
  fillable = FALSE,
  header = tags$head(tags$style(HTML("
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f7f9fc; }
    .hero { background: linear-gradient(135deg,#B22222 0%,#4682B4 100%);
            color:#fff; padding:36px; border-radius:16px; margin-bottom:24px;
            box-shadow: 0 8px 24px rgba(70,130,180,.25); }
    .hero h1 { font-weight:800; font-size: 32px; margin:0 0 8px 0; }
    .hero p  { opacity:.95; font-size:16px; margin:0 0 16px 0; }
    .badge-soft { display:inline-block; background:rgba(255,255,255,.18);
                  color:#fff; border:1px solid rgba(255,255,255,.4);
                  padding:4px 12px; border-radius:20px; margin:4px 4px 0 0;
                  font-size:12px; }
    .section-title { font-weight:800; color:#2c3e50; margin: 24px 0 12px 0;
                     border-left:5px solid #B22222; padding-left:12px; }
    .nav-tile { display:block; background:#fff; border-radius:12px;
                padding:18px 14px; text-align:center; cursor:pointer;
                border:2px solid #eef; transition:all .15s ease; }
    .nav-tile:hover { transform:translateY(-3px); border-color:#B22222;
                      box-shadow: 0 8px 20px rgba(178,34,34,.15); }
    .nav-tile .icon-wrap { font-size:26px; color:#B22222; margin-bottom:6px; }
    .nav-tile .label { font-weight:700; color:#2c3e50; font-size:14px; }
    .nav-tile .sub { color:#8896ab; font-size:12px; margin-top:2px; }
    .metric-mini { background:#fff; border-radius:10px; padding:14px;
                   text-align:center; border-top:3px solid #4682B4; }
    .metric-mini .val { font-size:24px; font-weight:800; color:#2c3e50; }
    .metric-mini .lbl { color:#8896ab; font-size:12px; text-transform:uppercase;
                        letter-spacing:.5px; margin-top:2px; }
    .driver-row { padding:8px 0; border-bottom:1px solid #eef; }
    .driver-row:last-child { border-bottom:none; }
    .driver-feature { font-weight:700; color:#2c3e50; }
    .driver-value { color:#8896ab; font-size:13px; }
    .driver-badge-up { background:#B22222; color:#fff; border-radius:12px;
                       padding:2px 8px; font-size:11px; font-weight:700; }
    .driver-badge-down { background:#4682B4; color:#fff; border-radius:12px;
                         padding:2px 8px; font-size:11px; font-weight:700; }
    .card { box-shadow: 0 2px 8px rgba(0,0,0,.04); border:none; }
  "))),
  
  ## === HOME ===
  nav_panel("Home", icon = icon("house"),
            div(class = "hero",
                h1("Online Payments Fraud Detection"),
                p("Real-time scoring · Explainable AI · Business-cost optimization"),
                div(
                  span(class="badge-soft", icon("bolt"), " Live predictions"),
                  span(class="badge-soft", icon("chart-line"), " SHAP + Break Down"),
                  span(class="badge-soft", icon("upload"), " Batch scoring"),
                  span(class="badge-soft", icon("sliders"), " Threshold tuning"))),
            layout_columns(fill = FALSE,
                           value_box("Transactions", comma(nrow(data_clean)),
                                     showcase = icon("credit-card"), theme = "primary"),
                           value_box("Fraud cases", sum(data_clean$is_fraud == "fraud"),
                                     showcase = icon("exclamation-triangle"), theme = "danger"),
                           value_box("Fraud rate",
                                     percent(mean(data_clean$is_fraud == "fraud"), accuracy = .01),
                                     showcase = icon("percent"), theme = "warning"),
                           value_box("Champion model", "Gradient Boosting",
                                     showcase = icon("brain"), theme = "info")),
            
            h3(class = "section-title", "Quick navigation"),
            layout_columns(
              div(class = "nav-tile",
                  onclick = "Shiny.setInputValue('nav_go','Predict',{priority:'event'})",
                  div(class="icon-wrap", icon("wand-magic-sparkles")),
                  div(class="label","Score a transaction"),
                  div(class="sub","Single-row prediction")),
              div(class = "nav-tile",
                  onclick = "Shiny.setInputValue('nav_go','Upload CSV',{priority:'event'})",
                  div(class="icon-wrap", icon("file-arrow-up")),
                  div(class="label","Upload a CSV"),
                  div(class="sub","Batch scoring + metrics")),
              div(class = "nav-tile",
                  onclick = "Shiny.setInputValue('nav_go','Explain XAI',{priority:'event'})",
                  div(class="icon-wrap", icon("magnifying-glass-chart")),
                  div(class="label","Explain a case"),
                  div(class="sub","SHAP + Break Down")),
              div(class = "nav-tile",
                  onclick = "Shiny.setInputValue('nav_go','Threshold',{priority:'event'})",
                  div(class="icon-wrap", icon("sliders")),
                  div(class="label","Tune threshold"),
                  div(class="sub","Cost optimization"))),
            
            h3(class = "section-title", "Population overview"),
            layout_columns(
              card(card_header(icon("chart-pie"), " Class imbalance"),
                   plotlyOutput("home_class", height = "260px")),
              card(card_header(icon("chart-column"), " Fraud rate by category"),
                   plotlyOutput("home_cat", height = "260px"))),
            layout_columns(
              card(card_header(icon("fire"), " Fraud-rate heatmap — hour × category"),
                   plotlyOutput("home_heatmap", height = "360px")),
              card(card_header(icon("chart-area"), " Amount distribution by class"),
                   plotlyOutput("home_amount_density", height = "360px"))),
            card(card_header(icon("list-ol"), " Top flagged test-set transactions"),
                 DTOutput("home_top_flagged"))
  ),
  
  ## === PREDICT ===
  nav_panel("Predict", icon = icon("wand-magic-sparkles"),
            layout_columns(col_widths = c(5, 7),
                           card(card_header(icon("edit"), " Transaction details"),
                                div(class="mb-2",
                                    actionButton("random_txn","Generate random transaction",
                                                 icon=icon("dice"),
                                                 class="btn-sm btn-outline-secondary w-100")),
                                numericInput("amt","Amount (USD)", 450, min=0, step=10),
                                sliderInput("hr","Hour (0–23)", 0, 23, 3),
                                selectInput("cat","Merchant category", choices=CAT_LEVELS,
                                            selected="Electronics"),
                                radioButtons("frn","Foreign transaction?",
                                             c("No"="no","Yes"="yes"), inline=TRUE),
                                radioButtons("mis","Location mismatch?",
                                             c("No"="no","Yes"="yes"), inline=TRUE),
                                sliderInput("trs","Device trust score (0–100)", 0, 100, 35),
                                sliderInput("vel","Velocity (txn in last 24h)", 0, 9, 6),
                                sliderInput("age","Cardholder age", 18, 69, 30),
                                actionButton("go","Score transaction",
                                             class="btn-primary w-100 mt-2", icon=icon("bolt"))),
                           tagList(
                             layout_columns(fill=FALSE,
                                            value_box("Fraud probability", textOutput("prob_txt"),
                                                      showcase=icon("chart-line"), theme="primary"),
                                            value_box("Risk band", textOutput("risk_txt"),
                                                      showcase=icon("traffic-light"), theme="danger"),
                                            value_box("Recommended action", textOutput("act_txt"),
                                                      showcase=icon("clipboard-check"), theme="success")),
                             card(card_header(icon("gauge-high"), " Probability gauge"),
                                  plotlyOutput("gauge", height = "240px")),
                             card(card_header(icon("list-check"),
                                              " Top risk drivers (Δ P(fraud) if feature reset)"),
                                  uiOutput("drivers_ui")))),
            layout_columns(
              card(card_header(icon("chart-bar"), " SHAP explanation"),
                   plotOutput("pred_shap", height="320px")),
              card(card_header(icon("water"), " Break Down explanation"),
                   plotOutput("pred_bd", height="320px"))),
            card(card_header(icon("location-crosshairs"),
                             " Where this transaction sits in the population"),
                 plotlyOutput("pred_position", height = "340px"))
  ),
  
  ## === UPLOAD ===
  nav_panel("Upload CSV", icon = icon("file-arrow-up"),
            layout_columns(col_widths = c(4, 8),
                           card(
                             card_header(icon("cloud-arrow-up"), " Upload credit.csv"),
                             fileInput("upload", NULL, accept=".csv"),
                             tags$hr(),
                             tags$b("Required"),
                             tags$ul(class="small",
                                     tags$li(tags$code("amount")), tags$li(tags$code("transaction_hour")),
                                     tags$li(tags$code("merchant_category")),
                                     tags$li(tags$code("foreign_transaction")),
                                     tags$li(tags$code("location_mismatch")),
                                     tags$li(tags$code("device_trust_score")),
                                     tags$li(tags$code("velocity_last_24h")),
                                     tags$li(tags$code("cardholder_age"))),
                             tags$b("Optional"), tags$ul(class="small",
                                                         tags$li(tags$code("is_fraud"), " — enables metrics")),
                             tags$hr(),
                             downloadButton("dl_scored_csv","Download scored CSV",
                                            class="btn-success w-100 mb-2"),
                             downloadButton("dl_scored_json","Download scored JSON",
                                            class="btn-outline-success w-100")),
                           tagList(
                             uiOutput("upload_status"),
                             layout_columns(fill=FALSE,
                                            value_box("Rows scored", textOutput("n_scored"),
                                                      showcase=icon("list-check"), theme="primary"),
                                            value_box("HIGH", textOutput("n_high"),
                                                      showcase=icon("exclamation-triangle"), theme="danger"),
                                            value_box("MEDIUM", textOutput("n_med"),
                                                      showcase=icon("circle-half-stroke"), theme="warning"),
                                            value_box("Auto-approved", textOutput("n_low"),
                                                      showcase=icon("check"), theme="success")),
                             conditionalPanel("output.has_labels",
                                              card(card_header(icon("bullseye"), " Performance on this file"),
                                                   layout_columns(
                                                     value_box("Accuracy",  textOutput("up_acc"),  theme="primary"),
                                                     value_box("Precision", textOutput("up_prec"), theme="info"),
                                                     value_box("Recall",    textOutput("up_rec"),  theme="success"),
                                                     value_box("F1",        textOutput("up_f1"),   theme="warning")),
                                                   DTOutput("up_cm"))),
                             layout_columns(
                               card(card_header(icon("chart-pie"), " Risk-band distribution"),
                                    plotlyOutput("up_donut", height="280px")),
                               card(card_header(icon("chart-area"), " Score distribution"),
                                    plotlyOutput("up_hist", height="280px"))),
                             card(card_header(icon("braille"), " Amount vs P(fraud)"),
                                  plotlyOutput("up_scatter", height = "320px")),
                             card(card_header(icon("table"), " Scored transactions"),
                                  layout_columns(
                                    selectInput("filter_band","Filter by risk band",
                                                c("All","HIGH","MEDIUM","LOW"), selected="All"),
                                    textOutput("filter_count")),
                                  DTOutput("up_tbl"))))
  ),
  
  ## === EXPLAIN ===
  nav_panel("Explain XAI", icon = icon("magnifying-glass-chart"),
            layout_columns(col_widths = c(4, 8),
                           card(card_header(icon("crosshairs"), " Pick a transaction"),
                                selectInput("row_pick", NULL, choices = NULL),
                                actionButton("go_xai","Explain", class="btn-primary w-100",
                                             icon=icon("play")),
                                tags$hr(), tableOutput("row_details")),
                           tagList(
                             card(card_header(icon("chart-bar"), " SHAP — contribution per feature"),
                                  plotOutput("shap_plot", height="360px")),
                             card(card_header(icon("water"), " Break Down — sequential narrative"),
                                  plotOutput("bd_plot", height="360px")))),
            card(card_header(icon("location-crosshairs"),
                             " Where this row sits in the population"),
                 plotlyOutput("row_position", height = "360px"))
  ),
  
  ## === THRESHOLD ===
  nav_panel("Threshold", icon = icon("sliders"),
            h3(class = "section-title", "Business-cost threshold tuning"),
            layout_columns(col_widths = c(4, 8),
                           card(
                             card_header(icon("sliders"), " Adjust decision threshold"),
                             sliderInput("threshold","Flag as fraud when P ≥",
                                         0.05, 0.95, 0.55, 0.01),
                             tags$hr(),
                             tags$b("Cost model"),
                             tags$ul(class="small",
                                     tags$li("False negative: ", tags$b("$500")),
                                     tags$li("False positive: ", tags$b("$25"))),
                             tags$hr(),
                             uiOutput("threshold_hint")),
                           tagList(
                             layout_columns(fill=FALSE,
                                            value_box("Precision", textOutput("th_prec"), theme="primary"),
                                            value_box("Recall",    textOutput("th_rec"),  theme="success"),
                                            value_box("F1",        textOutput("th_f1"),   theme="warning"),
                                            value_box("Batch cost", textOutput("th_cost"), theme="danger")),
                             card(card_header(icon("dollar-sign"), " Cost vs threshold"),
                                  plotlyOutput("th_cost_curve", height="280px")),
                             card(card_header(icon("chart-line"), " Precision / Recall trade-off"),
                                  plotlyOutput("th_pr_curve", height="280px")),
                             card(card_header(icon("table-cells"), " Confusion matrix at selected threshold"),
                                  DTOutput("th_cm"))))
  ),
  
  ## === MODELS ===
  nav_panel("Models", icon = icon("chart-simple"),
            h3(class = "section-title", "Model leaderboard"),
            layout_columns(
              card(card_header(icon("ranking-star"), " All models"),
                   DTOutput("tbl_models")),
              card(card_header(icon("chart-bar"), " PR-AUC by model"),
                   plotlyOutput("bar_models", height="300px"))),
            layout_columns(
              card(card_header(icon("chart-line"), " ROC curve"),
                   plotlyOutput("roc_plot", height="320px")),
              card(card_header(icon("chart-area"), " PR curve"),
                   plotlyOutput("pr_plot", height="320px"))),
            layout_columns(
              card(card_header(icon("star"), " Feature importance"),
                   plotlyOutput("feat_imp", height="360px")),
              card(card_header(icon("chart-pie"), " Confusion matrix at threshold 0.55"),
                   plotlyOutput("cm_heatmap", height="360px"))),
            h3(class = "section-title", "Champion model metrics"),
            layout_columns(fill=FALSE,
                           div(class="metric-mini",
                               div(class="val", textOutput("m_auc")), div(class="lbl","ROC-AUC")),
                           div(class="metric-mini",
                               div(class="val", textOutput("m_prauc")), div(class="lbl","PR-AUC")),
                           div(class="metric-mini",
                               div(class="val", textOutput("m_f1")), div(class="lbl","F1")),
                           div(class="metric-mini",
                               div(class="val", textOutput("m_recall")), div(class="lbl","Recall")))
  ),
  
  ## === ABOUT ===
  nav_panel("About", icon = icon("circle-info"),
            card(card_header(icon("brain"), " Method"),
                 tags$p(tags$b("Model:"), " Gradient Boosting (caret, 5-fold CV, up-sampling)."),
                 tags$p(tags$b("Features:"), " amount, hour, category, foreign flag,",
                        " mismatch, trust score, 24h velocity, age + 3 engineered booleans."),
                 tags$p(tags$b("Metrics:"), " PR-AUC, recall, F1 — accuracy is meaningless at 1.5%."),
                 tags$p(tags$b("XAI:"), " DALEX SHAP, Break Down, single-instance permutation."),
                 tags$p(tags$b("Business cost:"), " FN = $500, FP = $25, tuned in the Threshold tab.")),
            card(card_header(icon("robot"), " AI assistant (optional)"),
                 tags$p(class="text-muted small",
                        "Requires the ", tags$code("OPENAI_API_KEY"), " environment variable."),
                 textAreaInput("ai_q", NULL,
                               placeholder="e.g. Why might this transaction be risky?",
                               height="80px"),
                 actionButton("ai_ask","Ask", icon=icon("paper-plane"),
                              class="btn-outline-primary"),
                 tags$br(), tags$br(),
                 verbatimTextOutput("ai_ans")),
            card(tags$em("Dashboard by MAKINI CALEB — Task 6."))
  )
)

## ---- 5. Server -------------------------------------------------------------
server <- function(input, output, session) {
  
  observeEvent(input$nav_go, { nav_select("tabs", input$nav_go) })
  
  ## HOME
  output$home_class <- renderPlotly({
    d <- data_clean |> count(is_fraud)
    p <- ggplot(d, aes(is_fraud, n, fill = is_fraud)) +
      geom_col(width=.55, show.legend=FALSE) +
      geom_text(aes(label=comma(n)), vjust=-.4, fontface="bold") +
      scale_fill_manual(values=PAL) + scale_y_continuous(labels=comma) +
      labs(x=NULL, y="Transactions") + theme_minimal(base_size=12)
    ggplotly(p)
  })
  output$home_cat <- renderPlotly({
    d <- data_clean |> group_by(merchant_category) |>
      summarise(rate = mean(is_fraud=="fraud"), .groups="drop")
    p <- ggplot(d, aes(reorder(merchant_category, rate), rate, fill=rate)) +
      geom_col(show.legend=FALSE) +
      geom_text(aes(label=percent(rate,.1)), hjust=-.15, size=3.2) +
      scale_fill_gradient(low="#A9C4E0", high=PAL[["fraud"]]) +
      scale_y_continuous(labels=percent_format(.1),
                         expand=expansion(mult=c(0,.2))) +
      coord_flip() + labs(x=NULL, y="Fraud rate") + theme_minimal(base_size=12)
    ggplotly(p)
  })
  output$home_heatmap <- renderPlotly({
    d <- data_clean |>
      group_by(merchant_category, transaction_hour) |>
      summarise(n = n(), fraud_rate = mean(is_fraud == "fraud"), .groups = "drop") |>
      filter(n >= 20)
    p <- ggplot(d, aes(transaction_hour, merchant_category, fill = fraud_rate)) +
      geom_tile(colour = "white") +
      scale_fill_gradient(low = "#f7f9fc", high = PAL[["fraud"]],
                          labels = percent_format(0.1)) +
      scale_x_continuous(breaks = seq(0, 23, 3)) +
      labs(x = "Hour of day", y = NULL, fill = "Fraud rate") +
      theme_minimal(base_size = 12)
    ggplotly(p)
  })
  output$home_amount_density <- renderPlotly({
    d <- data_clean |> filter(amount > 0)
    p <- ggplot(d, aes(amount, fill = is_fraud, colour = is_fraud)) +
      geom_density(alpha = .35, linewidth = 1) +
      scale_x_log10(labels = dollar) +
      scale_fill_manual(values = PAL) +
      scale_colour_manual(values = PAL) +
      labs(x = "Amount (log)", y = "Density", fill = NULL, colour = NULL) +
      theme_minimal(base_size = 12) + theme(legend.position = "bottom")
    ggplotly(p)
  })
  output$home_top_flagged <- renderDT({
    df <- test_set |>
      mutate(p_fraud = test_probs,
             risk_band = cut(p_fraud, c(-Inf,.30,.55,Inf),
                             labels=c("LOW","MEDIUM","HIGH"))) |>
      arrange(desc(p_fraud)) |> head(10) |>
      select(p_fraud, risk_band, amount, merchant_category,
             device_trust_score, velocity_last_24h, is_fraud) |>
      mutate(amount = dollar(amount), p_fraud = round(p_fraud, 4))
    datatable(df, rownames = FALSE,
              colnames = c("P(fraud)","Risk","Amount","Category",
                           "Trust","Velocity","Actual"),
              options = list(dom = "t", pageLength = 10)) |>
      formatStyle("risk_band",
                  backgroundColor = styleEqual(c("LOW","MEDIUM","HIGH"),
                                               c("#D4EDDA","#FFF3CD","#F8D7DA")))
  })
  
  ## PREDICT
  observeEvent(input$random_txn, {
    tx <- demo_transaction()
    updateNumericInput(session, "amt", value=round(tx$amount, 2))
    updateSliderInput(session, "hr",  value=tx$hour)
    updateSelectInput(session, "cat", selected=tx$cat)
    updateRadioButtons(session, "frn", selected=tx$foreign)
    updateRadioButtons(session, "mis", selected=tx$mismatch)
    updateSliderInput(session, "trs", value=tx$trust)
    updateSliderInput(session, "vel", value=tx$velocity)
    updateSliderInput(session, "age", value=tx$age)
    showNotification(paste0("Random transaction loaded (actual: ", tx$actual, ")"),
                     type="message", duration=3)
  })
  pred_result <- eventReactive(input$go, {
    predict_one(input$amt, input$hr, input$cat, input$frn, input$mis,
                input$trs, input$vel, input$age)
  })
  output$prob_txt <- renderText({
    req(pred_result()); sprintf("%.2f%%", 100 * pred_result()$prob) })
  output$risk_txt <- renderText({
    req(pred_result()); p <- pred_result()$prob
    if (p < .30) "LOW" else if (p < .55) "MEDIUM" else "HIGH" })
  output$act_txt <- renderText({
    req(pred_result()); p <- pred_result()$prob
    if (p < .30) "Auto-approve"
    else if (p < .55) "Soft challenge / 2FA"
    else "Block & investigate" })
  output$gauge <- renderPlotly({
    req(pred_result())
    plot_ly(type="indicator", mode="gauge+number",
            value=round(100 * pred_result()$prob, 1),
            number=list(suffix="%"), title=list(text="P(fraud)"),
            gauge=list(
              axis=list(range=list(0,100), tickvals=c(0,30,55,100)),
              bar=list(color=PAL[["fraud"]]),
              steps=list(
                list(range=c(0,30),  color="#D4EDDA"),
                list(range=c(30,55), color="#FFF3CD"),
                list(range=c(55,100),color="#F8D7DA")),
              threshold=list(line=list(color="#2c3e50", width=4),
                             thickness=0.75, value=55))) |>
      config(displayModeBar=FALSE)
  })
  output$drivers_ui <- renderUI({
    req(pred_result())
    drv <- compute_local_drivers(pred_result()$row, x_test) |> head(5)
    tagList(
      tags$p(class="text-muted small",
             "Δ = change in P(fraud) if this feature were set to the population median/mode."),
      lapply(seq_len(nrow(drv)), function(i) {
        row <- drv[i, ]; up <- row$delta > 0
        div(class="driver-row d-flex justify-content-between align-items-center",
            div(div(class="driver-feature", row$feature),
                div(class="driver-value","value: ", row$value)),
            div(span(class=if (up) "driver-badge-up" else "driver-badge-down",
                     sprintf("%s%.2f%%", if (up) "+" else "", row$pct * 100)),
                span(class="ms-2 small text-muted",
                     if (up) "risk ↑" else "risk ↓")))
      })
    )
  })
  output$pred_shap <- renderPlot({
    req(pred_result())
    withProgress(message="Computing SHAP…", value=.4, {
      p <- DALEX::predict_parts(explainer,
                                new_observation=pred_result()$row,
                                type="shap", B=15)
    })
    plot(p) + labs(title="SHAP — feature contributions")
  })
  output$pred_bd <- renderPlot({
    req(pred_result())
    plot(DALEX::predict_parts(explainer,
                              new_observation=pred_result()$row,
                              type="break_down"))
  })
  output$pred_position <- renderPlotly({
    req(pred_result())
    row <- pred_result()$row
    feats <- c("amount","transaction_hour","device_trust_score",
               "velocity_last_24h","cardholder_age")
    pcts <- sapply(feats, function(f)
      pct_rank(x_test[[f]], as.numeric(row[[f]])))
    d <- tibble(feature = feats, percentile = pcts)
    p <- ggplot(d, aes(reorder(feature, percentile), percentile)) +
      geom_col(fill = PAL[["fraud"]], width = .55) +
      geom_vline(xintercept = 50, linetype="dashed", colour="grey40") +
      geom_text(aes(label = paste0(percentile, "%")),
                hjust = -0.2, size = 3.4) + coord_flip() +
      scale_y_continuous(limits = c(0, 110),
                         breaks = c(0, 25, 50, 75, 100),
                         labels = function(x) paste0(x, "th")) +
      labs(x = NULL, y = "Percentile (population)",
           title = "How this transaction compares to all 9,999 cases") +
      theme_minimal(base_size = 12)
    ggplotly(p)
  })
  
  ## UPLOAD
  upload_result <- reactive({
    req(input$upload)
    prep_and_score(read.csv(input$upload$datapath, stringsAsFactors = FALSE))
  })
  output$upload_status <- renderUI({
    req(upload_result()); r <- upload_result()
    cls <- if (nrow(r$df) > 0) "alert-info" else "alert-warning"
    extra <- if (r$n_dropped > 0)
      paste0(" ", r$n_dropped, " row(s) removed (amount ≤ 0 or unknown category).")
    tags$div(class = paste("alert", cls), icon("circle-info"),
             if (nrow(r$df) > 0)
               paste0(" Scored ", comma(nrow(r$df)), " rows.", extra)
             else " No valid rows after cleaning.")
  })
  output$n_scored <- renderText({ req(upload_result()); comma(nrow(upload_result()$df)) })
  output$n_high   <- renderText({ req(upload_result()); comma(sum(upload_result()$df$risk_band == "HIGH")) })
  output$n_med    <- renderText({ req(upload_result()); comma(sum(upload_result()$df$risk_band == "MEDIUM")) })
  output$n_low    <- renderText({ req(upload_result()); comma(sum(upload_result()$df$risk_band == "LOW")) })
  output$has_labels <- reactive({ req(upload_result()); upload_result()$has_labels })
  outputOptions(output, "has_labels", suspendWhenHidden = FALSE)
  upload_metrics <- reactive({
    r <- upload_result(); if (!r$has_labels) return(NULL)
    pred <- factor(ifelse(r$df$p_fraud >= .55, "fraud", "legit"),
                   levels = c("legit","fraud"))
    list(cm = caret::confusionMatrix(pred, r$df$actual, positive = "fraud"))
  })
  output$up_acc  <- renderText({ m <- upload_metrics(); if (is.null(m)) "—" else sprintf("%.3f", m$cm$overall["Accuracy"]) })
  output$up_prec <- renderText({ m <- upload_metrics(); if (is.null(m)) "—" else sprintf("%.3f", m$cm$byClass["Precision"]) })
  output$up_rec  <- renderText({ m <- upload_metrics(); if (is.null(m)) "—" else sprintf("%.3f", m$cm$byClass["Sensitivity"]) })
  output$up_f1   <- renderText({ m <- upload_metrics(); if (is.null(m)) "—" else sprintf("%.3f", m$cm$byClass["F1"]) })
  output$up_cm <- renderDT({
    m <- upload_metrics(); req(m)
    tab <- as.data.frame.matrix(m$cm$table)
    tab <- cbind(Predicted = rownames(tab), tab)
    datatable(tab, rownames = FALSE, options = list(dom = "t", pageLength = 5)) |>
      formatStyle(columns = 2:3, fontWeight = "bold")
  })
  output$up_donut <- renderPlotly({
    req(upload_result())
    d <- upload_result()$df |> count(risk_band)
    plot_ly(d, labels = ~risk_band, values = ~n, type = "pie", hole = 0.55,
            marker = list(colors = c("#D4EDDA","#FFF3CD","#F8D7DA")),
            textinfo = "label+percent") |>
      layout(showlegend = FALSE, margin = list(t = 0, b = 0))
  })
  output$up_hist <- renderPlotly({
    req(upload_result()); df <- upload_result()$df
    p <- ggplot(df, aes(p_fraud, fill = risk_band)) +
      geom_histogram(bins = 40, color = "white", alpha = .85) +
      geom_vline(xintercept = c(.30, .55), linetype = "dashed", colour = "grey40") +
      scale_fill_manual(values = c("LOW"="#D4EDDA","MEDIUM"="#FFF3CD","HIGH"="#F8D7DA")) +
      labs(x = "P(fraud)", y = "Count", fill = "Risk") +
      theme_minimal(base_size = 12) + theme(legend.position = "bottom")
    ggplotly(p)
  })
  output$up_scatter <- renderPlotly({
    req(upload_result()); df <- upload_result()$df
    p <- ggplot(df, aes(amount, p_fraud, colour = risk_band,
                        text = paste0("Amount: ", dollar(amount),
                                      "\nP(fraud): ", round(p_fraud, 4)))) +
      geom_jitter(alpha = .55, width = .03, height = .01, size = 1.1) +
      scale_x_log10(labels = dollar) +
      scale_colour_manual(values = c("LOW"="#27ae60","MEDIUM"="#e67e22","HIGH"="#B22222")) +
      geom_hline(yintercept = c(.30, .55), linetype = "dashed", colour = "grey50") +
      labs(x = "Amount (log)", y = "P(fraud)", colour = "Risk") +
      theme_minimal(base_size = 12) + theme(legend.position = "bottom")
    ggplotly(p, tooltip = "text")
  })
  filtered_upload <- reactive({
    req(upload_result()); df <- upload_result()$df
    if ((input$filter_band %||% "All") != "All")
      df <- df[df$risk_band == input$filter_band, , drop = FALSE]
    df
  })
  output$filter_count <- renderText({
    req(filtered_upload())
    paste0("Showing ", comma(nrow(filtered_upload())), " of ",
           comma(nrow(upload_result()$df)), " rows")
  })
  output$up_tbl <- renderDT({
    req(filtered_upload()); df <- filtered_upload()
    cols <- c("amount","merchant_category","p_fraud","risk_band",
              "recommended_action",
              if (upload_result()$has_labels) "actual")
    df |> arrange(desc(p_fraud)) |> select(all_of(cols)) |>
      datatable(options = list(pageLength = 10, scrollX = TRUE)) |>
      formatRound("p_fraud", 4) |>
      formatCurrency("amount", currency = "$", digits = 2) |>
      formatStyle("risk_band",
                  backgroundColor = styleEqual(c("LOW","MEDIUM","HIGH"),
                                               c("#D4EDDA","#FFF3CD","#F8D7DA")))
  })
  output$dl_scored_csv <- downloadHandler(
    filename = function() paste0("scored_", Sys.Date(), ".csv"),
    content  = function(f) write.csv(upload_result()$df, f, row.names = FALSE))
  output$dl_scored_json <- downloadHandler(
    filename = function() paste0("scored_", Sys.Date(), ".json"),
    content  = function(f) {
      req(upload_result())
      if (!requireNamespace("jsonlite", quietly = TRUE)) { writeLines("[]", f); return() }
      jsonlite::write_json(upload_result()$df, f, pretty = TRUE)
    })
  
  ## EXPLAIN
  top_idx <- order(test_probs, decreasing = TRUE)[1:25]
  updateSelectInput(session, "row_pick",
                    choices = setNames(top_idx,
                                       paste0("Row ", top_idx, " — P=", round(test_probs[top_idx], 3),
                                              ifelse(y_test[top_idx]==1, " [fraud]", " [legit]"))),
                    selected = top_idx[1])
  output$row_details <- renderTable({
    idx <- as.integer(input$row_pick)
    tibble(Field = c("Row","P(fraud)","Actual","Amount","Hour","Velocity",
                     "Trust","Category","Foreign","Mismatch"),
           Value = c(idx, round(test_probs[idx], 3),
                     as.character(test_set$is_fraud[idx]),
                     dollar(test_set$amount[idx]),
                     test_set$transaction_hour[idx],
                     test_set$velocity_last_24h[idx],
                     test_set$device_trust_score[idx],
                     as.character(test_set$merchant_category[idx]),
                     as.character(test_set$foreign_transaction[idx]),
                     as.character(test_set$location_mismatch[idx])))
  }, striped = TRUE, bordered = TRUE)
  shap_res <- eventReactive(input$go_xai, {
    idx <- as.integer(input$row_pick)
    withProgress(message="SHAP…", value=.4,
                 DALEX::predict_parts(explainer,
                                      new_observation = x_test[idx, , drop=FALSE],
                                      type="shap", B=15))
  })
  output$shap_plot <- renderPlot({
    req(shap_res())
    plot(shap_res()) +
      labs(title=sprintf("Row %s — P=%.3f", input$row_pick,
                         test_probs[as.integer(input$row_pick)]))
  })
  bd_res <- eventReactive(input$go_xai, {
    idx <- as.integer(input$row_pick)
    DALEX::predict_parts(explainer,
                         new_observation = x_test[idx, , drop=FALSE],
                         type = "break_down")
  })
  output$bd_plot <- renderPlot({ req(bd_res()); plot(bd_res()) })
  output$row_position <- renderPlotly({
    req(input$go_xai > 0)
    idx <- as.integer(input$row_pick)
    feats <- c("amount","transaction_hour","device_trust_score",
               "velocity_last_24h","cardholder_age")
    pcts <- sapply(feats, function(f)
      pct_rank(x_test[[f]], as.numeric(x_test[idx, f])))
    d <- tibble(feature = feats, percentile = pcts)
    p <- ggplot(d, aes(reorder(feature, percentile), percentile)) +
      geom_col(fill = PAL[["fraud"]], width = .55) +
      geom_vline(xintercept = 50, linetype="dashed", colour="grey40") +
      geom_text(aes(label = paste0(percentile, "%")),
                hjust = -0.2, size = 3.4) + coord_flip() +
      scale_y_continuous(limits = c(0, 110),
                         breaks = c(0, 25, 50, 75, 100),
                         labels = function(x) paste0(x, "th")) +
      labs(x = NULL, y = "Percentile (population)",
           title = paste0("Row ", idx, " vs all 9,999 test cases")) +
      theme_minimal(base_size = 12)
    ggplotly(p)
  })
  
  ## THRESHOLD
  th_metrics <- reactive({
    th <- input$threshold %||% 0.55
    pred <- factor(ifelse(test_probs >= th, "fraud", "legit"),
                   levels = c("legit","fraud"))
    cm <- caret::confusionMatrix(pred, test_set$is_fraud, positive = "fraud")
    tp <- cm$table["fraud","fraud"]; fp <- cm$table["fraud","legit"]
    fn <- cm$table["legit","fraud"]; tn <- cm$table["legit","legit"]
    list(cm = cm, tp = tp, fp = fp, fn = fn, tn = tn,
         cost = fn * COST_FN + fp * COST_FP)
  })
  output$th_prec <- renderText({ sprintf("%.3f", th_metrics()$cm$byClass["Precision"]) })
  output$th_rec  <- renderText({ sprintf("%.3f", th_metrics()$cm$byClass["Sensitivity"]) })
  output$th_f1   <- renderText({ sprintf("%.3f", th_metrics()$cm$byClass["F1"]) })
  output$th_cost <- renderText({ dollar(th_metrics()$cost) })
  output$threshold_hint <- renderUI({
    m <- th_metrics()
    tags$div(class="alert alert-light small", icon("info-circle"),
             sprintf(" At threshold %.2f: TP=%d, FP=%d, FN=%d, TN=%d → cost $%s",
                     input$threshold %||% 0.55, m$tp, m$fp, m$fn, m$tn, comma(m$cost)))
  })
  th_curve <- reactive({
    tibble(threshold = seq(0.05, 0.95, 0.01)) |> rowwise() |>
      mutate(
        tp = sum(test_probs >= threshold & y_test == 1),
        fp = sum(test_probs >= threshold & y_test == 0),
        fn = sum(test_probs <  threshold & y_test == 1),
        precision = ifelse(tp + fp > 0, tp/(tp+fp), NA),
        recall    = tp/(tp+fn),
        cost      = fn * COST_FN + fp * COST_FP) |> ungroup()
  })
  output$th_cost_curve <- renderPlotly({
    d <- th_curve()
    p <- ggplot(d, aes(threshold, cost)) +
      geom_line(colour = PAL[["fraud"]], linewidth = 1.2) +
      geom_vline(xintercept = input$threshold %||% 0.55,
                 linetype="dashed", colour="grey30") +
      geom_point(data = d[which.min(d$cost), ], colour = "black", size = 3) +
      scale_y_continuous(labels = dollar) +
      labs(x = "Threshold", y = "Batch cost") + theme_minimal(base_size = 12)
    ggplotly(p)
  })
  output$th_pr_curve <- renderPlotly({
    d <- th_curve() |>
      select(threshold, Precision = precision, Recall = recall) |>
      pivot_longer(-threshold)
    p <- ggplot(d, aes(threshold, value, colour = name)) +
      geom_line(linewidth = 1.1) +
      geom_vline(xintercept = input$threshold %||% 0.55,
                 linetype="dashed", colour="grey30") +
      labs(x = "Threshold", y = "Value", colour = NULL) +
      theme_minimal(base_size = 12) + theme(legend.position = "bottom")
    ggplotly(p)
  })
  output$th_cm <- renderDT({
    m <- th_metrics()
    tab <- as.data.frame.matrix(m$cm$table)
    tab <- cbind(Predicted = rownames(tab), tab)
    datatable(tab, rownames = FALSE, options = list(dom = "t")) |>
      formatStyle(columns = 2:3, fontWeight = "bold")
  })
  
  ## MODELS
  output$tbl_models <- renderDT({
    datatable(comparison, rownames = FALSE,
              options = list(dom = "t", pageLength = 10)) |>
      formatStyle("PR_AUC",
                  background = styleColorBar(comparison$PR_AUC, "#B22222"),
                  backgroundSize = "100% 90%", backgroundRepeat = "no-repeat",
                  backgroundPosition = "center")
  })
  output$bar_models <- renderPlotly({
    p <- ggplot(comparison, aes(reorder(Model, PR_AUC), PR_AUC, fill = Model)) +
      geom_col(show.legend = FALSE) +
      geom_text(aes(label = sprintf("%.3f", PR_AUC)), hjust = -.15, size = 3.2) +
      coord_flip() + scale_y_continuous(expand = expansion(mult = c(0,.15))) +
      labs(x = NULL, y = "PR-AUC") + theme_minimal(base_size = 12)
    ggplotly(p)
  })
  output$roc_plot <- renderPlotly({
    r <- pROC::roc(test_set$is_fraud, test_probs, quiet = TRUE,
                   levels = c("legit","fraud"), direction = "<")
    df <- tibble(fpr = 1 - r$specificities, tpr = r$sensitivities) |> arrange(fpr)
    p <- ggplot(df, aes(fpr, tpr)) +
      geom_abline(linetype="dashed", colour="grey60") +
      geom_path(colour = PAL[["fraud"]], linewidth = 1.1) + coord_equal() +
      labs(x = "FPR", y = "TPR",
           title = sprintf("AUC = %.3f", as.numeric(pROC::auc(r)))) +
      theme_minimal(base_size = 12)
    ggplotly(p)
  })
  output$pr_plot <- renderPlotly({
    df <- yardstick::pr_curve(tibble(truth = test_set$is_fraud, prob = test_probs),
                              truth, prob, event_level = "second")
    pauc <- yardstick::pr_auc_vec(test_set$is_fraud, test_probs, event_level = "second")
    p <- ggplot(df, aes(recall, precision)) +
      geom_path(colour = PAL[["fraud"]], linewidth = 1.1) +
      labs(x = "Recall", y = "Precision",
           title = sprintf("PR-AUC = %.3f", pauc)) + theme_minimal(base_size = 12)
    ggplotly(p)
  })
  output$feat_imp <- renderPlotly({
    if (nrow(gbm_imp) == 0)
      return(plot_ly() |> add_annotations(text="Not available", x=.5, y=.5, showarrow=FALSE))
    d <- gbm_imp |> head(10)
    p <- ggplot(d, aes(reorder(Feature, Overall), Overall)) +
      geom_col(fill = PAL[["fraud"]], width = .65) +
      geom_text(aes(label = round(Overall, 1)), hjust = -.15, size = 3.3) +
      coord_flip() + scale_y_continuous(expand = expansion(mult = c(0, .2))) +
      labs(x = NULL, y = "Relative influence (caret varImp)") +
      theme_minimal(base_size = 12)
    ggplotly(p)
  })
  output$cm_heatmap <- renderPlotly({
    pred <- factor(ifelse(test_probs >= 0.55, "fraud", "legit"),
                   levels = c("legit","fraud"))
    d <- as.data.frame(table(Predicted = pred, Actual = test_set$is_fraud))
    p <- ggplot(d, aes(Predicted, Actual, fill = Freq)) +
      geom_tile(colour = "white", linewidth = 1.2) +
      geom_text(aes(label = comma(Freq)), colour = "white",
                size = 7, fontface = "bold") +
      scale_fill_gradient(low = "#7BA8D9", high = PAL[["fraud"]]) +
      labs(x = "Predicted", y = "Actual",
           title = "Confusion matrix at threshold 0.55") +
      theme_minimal(base_size = 12) + theme(legend.position = "none")
    ggplotly(p) |> config(displayModeBar = FALSE)
  })
  output$m_auc    <- renderText({ as.character(round(comparison$ROC_AUC[comparison$Model=="Gradient Boosting"],3)) })
  output$m_prauc  <- renderText({ as.character(round(comparison$PR_AUC[comparison$Model=="Gradient Boosting"],3)) })
  output$m_f1     <- renderText({ as.character(round(comparison$F1[comparison$Model=="Gradient Boosting"],3)) })
  output$m_recall <- renderText({ as.character(round(comparison$Recall[comparison$Model=="Gradient Boosting"],3)) })
  
  ## AI ASSISTANT
  observeEvent(input$ai_ask, {
    key <- Sys.getenv("OPENAI_API_KEY")
    if (!nzchar(key)) {
      output$ai_ans <- renderText(
        paste0("AI assistant not configured (set OPENAI_API_KEY).\n\n",
               "Prompt that would be sent:\nP(fraud)=",
               if (is.null(pred_result())) "N/A" else paste0(round(100*pred_result()$prob,2),"%"),
               "\nQuestion: ", input$ai_q)); return()
    }
    if (!requireNamespace("httr", quietly=TRUE) ||
        !requireNamespace("jsonlite", quietly=TRUE)) {
      output$ai_ans <- renderText("httr + jsonlite needed."); return()
    }
    withProgress(message="Asking AI…", value=.5, {
      body <- list(model = "gpt-4o-mini",
                   messages = list(list(role = "user",
                                        content = paste0("You are a fraud analyst. Score: ",
                                                         if (is.null(pred_result())) "N/A" else paste0(round(100*pred_result()$prob,2),"%"),
                                                         ". Context: ",
                                                         if (is.null(pred_result())) "—" else jsonlite::toJSON(pred_result()$row, auto_unbox=TRUE),
                                                         ". Answer in 3 concise sentences: ", input$ai_q))))
      res <- tryCatch(httr::POST("https://api.openai.com/v1/chat/completions",
                                 httr::add_headers(Authorization=paste("Bearer", key)),
                                 httr::content_type_json(),
                                 body = jsonlite::toJSON(body, auto_unbox=TRUE),
                                 encode="raw"), error=function(e) NULL)
      if (is.null(res) || httr::status_code(res) != 200) {
        output$ai_ans <- renderText("AI request failed."); return()
      }
      output$ai_ans <- renderText(httr::content(res)$choices[[1]]$message$content)
    })
  })
}

shinyApp(ui, server)