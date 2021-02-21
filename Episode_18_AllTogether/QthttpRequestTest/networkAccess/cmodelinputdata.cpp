#include "cmodelinputdata.h"

CModelInputData::CModelInputData(quint16 historySize)
    :historySize(historySize)
{
    if (historySize == 0)
    {
        historySize = 1;
    }

    for(quint16 i=0; i<historySize; ++i)
    {
        this->pos.push_back(QJsonArray({0.0,0.0}));
        this->price.push_back(QJsonArray({0.0,0.0,0.0,0.0,0.0}));
    }
}


QJsonObject CModelInputData::toJsonObj() const
{
    QJsonObject rootJson;
    rootJson["price"] = this->price;
    rootJson["pos"] = this->pos;
    return rootJson;
}

QByteArray CModelInputData::toJson() const
{
    return QJsonDocument(this->toJsonObj()).toJson();
}

void CModelInputData::addObservation(QJsonArray pos, QJsonArray price)
{
    if((5u == price.size()) && (2u == pos.size()))
    {
        this->pos.pop_front();
        this->pos.push_back(pos);
        this->price.pop_front();
        this->price.push_back(price);
    }
}
