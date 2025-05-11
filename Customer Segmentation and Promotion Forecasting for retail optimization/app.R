# Load necessary libraries
library(shiny)
library(ggplot2)
library(dplyr)
library(cluster)
library(factoextra)
library(caret)
library(forecast)

# Load dataset
customer_data <- read.csv("customer_data.csv")

# Data Preprocessing: Remove missing values
customer_data <- na.omit(customer_data)

# UI: Define the dashboard layout
ui <- fluidPage(
  titlePanel("Customer Analytics Dashboard"),

  sidebarLayout(
    sidebarPanel(
      selectInput("xvar", "Select X-axis Variable:", choices = names(customer_data), selected = "Income"),
      selectInput("yvar", "Select Y-axis Variable:", choices = names(customer_data), selected = "MntWines"),
      numericInput("clusters", "Number of Clusters:", 3, min = 2, max = 10),
      actionButton("runClustering", "Run Clustering"),
      hr(),
      actionButton("runPrediction", "Run Promotion Prediction"),
      actionButton("runForecasting", "Run Sales Forecast")
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("Customer Clustering", plotOutput("clusterPlot")),
        tabPanel("Promotion Prediction", tableOutput("predictionResults")),
        tabPanel("Sales Forecasting", plotOutput("forecastPlot"))
      )
    )
  )
)

# Server Logic
server <- function(input, output) {

  # Perform K-Means Clustering
  observeEvent(input$runClustering, {
    req(input$clusters)

    # Select relevant columns for clustering
    customer_features <- customer_data %>%
      select(Income, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds)

    # Standardize data
    customer_scaled <- scale(customer_features)

    # Ensure cluster number is valid
    max_clusters <- min(nrow(customer_scaled), input$clusters)

    # Run K-Means
    k_clusters <- kmeans(customer_scaled, centers = max_clusters, nstart = 25)
    customer_data$Cluster <- as.factor(k_clusters$cluster)

    # Create Cluster Plot
    output$clusterPlot <- renderPlot({
      fviz_cluster(k_clusters, data = customer_scaled) +
        ggtitle("Customer Segmentation (K-Means Clustering)")
    })
  })

  # Promotion Effectiveness Prediction (Logistic Regression)
  observeEvent(input$runPrediction, {
    # Check if necessary columns exist
    if (!("AcceptedCmp1" %in% colnames(customer_data))) {
      output$predictionResults <- renderTable({ data.frame(Error = "No Promotion Data Found!") })
      return()
    }

    # Split Data (Train/Test)
    set.seed(123)
    trainIndex <- createDataPartition(customer_data$AcceptedCmp1, p = 0.7, list = FALSE)
    trainData <- customer_data[trainIndex, ]
    testData <- customer_data[-trainIndex, ]

    # Logistic Regression Model
    model <- glm(AcceptedCmp1 ~ Income + MntWines + MntFruits + MntMeatProducts, data = trainData, family = binomial)

    # Predict on Test Data
    testData$Predicted <- predict(model, testData, type = "response")
    testData$PredictedClass <- ifelse(testData$Predicted > 0.5, 1, 0)

    # Show Prediction Results
    output$predictionResults <- renderTable({
      testData %>%
        select(ID, Income, MntWines, PredictedClass) %>%
        head(10)
    })
  })

  # Sales Forecasting using ARIMA
  observeEvent(input$runForecasting, {
    if (!("TotalSales" %in% colnames(customer_data))) {
      output$forecastPlot <- renderPlot({ plot(0, 0, main = "No Sales Data Available") })
      return()
    }

    # Aggregate sales by month
    sales_ts <- ts(customer_data$TotalSales, frequency = 12, start = c(2012, 1))

    # Fit ARIMA model
    model <- auto.arima(sales_ts)

    # Forecast next 12 months
    forecasted_values <- forecast(model, h = 12)

    # Show Forecast Plot
    output$forecastPlot <- renderPlot({
      plot(forecasted_values, main = "Sales Forecast for Next 12 Months", col = "blue")
    })
  })
}

# Run the Shiny App
shinyApp(ui = ui, server = server)
