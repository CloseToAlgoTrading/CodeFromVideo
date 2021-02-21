#ifndef CMODELINPUTDATA_H
#define CMODELINPUTDATA_H

#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

class CModelInputData
{
public:
    CModelInputData(quint16 historySize);
    QJsonObject toJsonObj() const;
    QByteArray toJson() const;

    void addObservation(QJsonArray pos, QJsonArray price);

private:
    quint16 historySize;
    QJsonArray pos;
    QJsonArray price;

};

#endif // CMODELINPUTDATA_H
