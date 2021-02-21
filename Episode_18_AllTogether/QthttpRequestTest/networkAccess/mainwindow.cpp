#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QNetworkReply>
#include <QString>
#include <QHttp2Configuration>

#include <QJsonObject>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTextStream>
#include "cmodelinputdata.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    // Connect NetworkAccess
    connect(&manager, &QNetworkAccessManager::finished, this, &MainWindow::onManagerFinished);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    ui->textEdit->append("Send request");
    CModelInputData test_data(3);

    test_data.addObservation({1.0,0.0}, {0.11,0.12,0.11,0.23,0.444});
    test_data.addObservation({0.0,1.0}, {1.11,1.12,1.11,1.23,1.444});

    QByteArray ba = test_data.toJson();
    ui->textEdit->append(ba);

    QNetworkRequest request = QNetworkRequest(QUrl("http://localhost:49154/predict"));
    request.setRawHeader("Content-Type","application/json");
    request.setRawHeader("app_id", "user1");

    this->manager.post(request, ba);
}

void MainWindow::onManagerFinished(QNetworkReply *reply)
{
    ui->textEdit->append("Reply received");
    if (reply->error() != QNetworkReply::NoError)
    {
        qDebug() << reply->errorString();
        reply->deleteLater();
        return;
    }

    QByteArray  content = reply->readAll();

    if(reply->hasRawHeader("app_id") == true)
    {
        QByteArray app_id = reply->rawHeader("app_id");
        if(0 == QString::compare(app_id, "user1", Qt::CaseInsensitive))
        {
            ui->textEdit->append("received prediction json response");
        }
        else
        {
            ui->textEdit->append("received reload response");
        }
    }

    ui->textEdit->append(content);

    reply->deleteLater();
}


