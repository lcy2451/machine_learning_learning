from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":  
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 

    cap_cross = X["cap-shape"].astype(str) + "_" + X["cap-surface"] + "_" + X["cap-color"]
    # print(cap_cross)

    # 创建一个包含分类变量的 Series
    cap_encoder = OneHotEncoder()
    # 学习规则
    cap_encoder.fit(cap_cross.to_frame()) # 90个输入特征
    # 转换
    cap_transformed = cap_encoder.transform(cap_cross.to_frame()) 

    # print(len(cap_encoder.categories_[0]))
    # 创建一个包含分类变量的 Series
    cap_shape_encoder = OneHotEncoder()

    # 学习规则
    cap_shape_encoder.fit(X[["cap-shape"]]) # 6个特征

    cap_shape_transformed = cap_shape_encoder.transform(X[["cap-shape"]]) 

    # 创建一个包含分类变量的 Series
    cap_surface_encoder = OneHotEncoder()

    # 学习规则
    cap_surface_encoder.fit(X[["cap-surface"]]) # 4个特征

    # 转换
    cap_surface_transformed = cap_surface_encoder.transform(X[["cap-surface"]]) 
    # print(len(cap_surface_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    cap_color_encoder = OneHotEncoder()

    # 学习规则
    cap_color_encoder.fit(X[["cap-color"]]) # 10个特征

    # 转换
    cap_color_transformed = cap_color_encoder.transform(X[["cap-color"]]) 

    # print(len(cap_color_encoder.categories_[0]))

    gill_cross = X["gill-spacing"].astype(str) + "_" + X["gill-attachment"] + "_" + X["gill-size"] + "_" \
                + X["gill-color"] 

    # 创建一个包含分类变量的 Series
    gill_encoder = OneHotEncoder()
    # 学习规则
    gill_encoder.fit(gill_cross.to_frame()) # 34个输入特征
    # 转换
    gill_transformed = gill_encoder.transform(gill_cross.to_frame()) 

    # print(len(gill_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    gill_spacing_encoder = OneHotEncoder()

    # 学习规则
    gill_spacing_encoder.fit(X[["gill-spacing"]]) # 2个特征 ， 文档里有三个

    # 转换
    gill_spacing_transformed = gill_spacing_encoder.transform(X[["gill-spacing"]]) 

    # print(gill_spacing_transformed.shape)

    # 创建一个包含分类变量的 Series
    gill_attachment_encoder = OneHotEncoder()

    # 学习规则
    gill_attachment_encoder.fit(X[["gill-attachment"]]) # 2个特征 ， 文档里有四个

    # 转换
    gill_attachment_transformed = gill_attachment_encoder.transform(X[["gill-attachment"]]) 

    # print("gill_attachment_encoder ", len(gill_attachment_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    gill_size_encoder = OneHotEncoder()

    # 学习规则
    gill_size_encoder.fit(X[["gill-size"]]) # 2个特征 

    # 转换
    gill_size_transformed = gill_size_encoder.transform(X[["gill-size"]]) 

    # print("gill_size_encoder ", len(gill_size_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    gill_color_encoder = OneHotEncoder()

    # 学习规则
    gill_color_encoder.fit(X[["gill-color"]]) # 12个特征 

    # 转换
    gill_color_transformed = gill_color_encoder.transform(X[["gill-color"]]) 

    # print("gill_color_encoder ", len(gill_color_encoder.categories_[0]))

    stalk_cross = X["stalk-shape"].astype(str) + "_" + X["stalk-root"] + "_" + X["stalk-surface-above-ring"] + "_" \
                + X["stalk-surface-below-ring"] + "_" + X["stalk-color-above-ring"] + "_" + X["stalk-color-below-ring"]

    # 创建一个包含分类变量的 Series
    stalk_encoder = OneHotEncoder()
    # 学习规则
    stalk_encoder.fit(stalk_cross.to_frame()) # 37个输入特征
    # 转换
    stalk_transformed = stalk_encoder.transform(stalk_cross.to_frame()) 

    # print("stalk_encoder ", len(stalk_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    stalk_shape_encoder = OneHotEncoder()

    # 学习规则
    stalk_shape_encoder.fit(X[["stalk-shape"]]) # 2个特征 

    # 转换
    stalk_shape_transformed = stalk_shape_encoder.transform(X[["stalk-shape"]]) 

    # print("stalk_shape_encoder ", len(stalk_shape_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    stalk_root_encoder = OneHotEncoder()

    # 学习规则
    stalk_root_encoder.fit(X[["stalk-root"]]) # 5个特征 

    # 转换
    stalk_root_transformed = stalk_root_encoder.transform(X[["stalk-root"]]) 

    # print("stalk_root_encoder ", len(stalk_root_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    stalk_surface_above_ring_encoder = OneHotEncoder()

    # 学习规则
    stalk_surface_above_ring_encoder.fit(X[["stalk-surface-above-ring"]]) # 4个特征 

    # 转换
    stalk_surface_above_ring_transformed = stalk_surface_above_ring_encoder.transform(X[["stalk-surface-above-ring"]]) 

    # print("stalk_surface_above_ring_encoder ", len(stalk_surface_above_ring_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    stalk_surface_below_ringencoder = OneHotEncoder()

    # 学习规则
    stalk_surface_below_ringencoder.fit(X[["stalk-surface-below-ring"]]) # 4个特征 

    # 转换
    stalk_surface_below_ring_transformed = stalk_surface_below_ringencoder.transform(X[["stalk-surface-below-ring"]]) 

    # print("stalk_surface_below_ringencoder ", len(stalk_surface_below_ringencoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    stalk_color_above_ring_encoder = OneHotEncoder()

    # 学习规则
    stalk_color_above_ring_encoder.fit(X[["stalk-color-above-ring"]]) # 9个特征 

    # 转换
    stalk_color_above_ring_transformed = stalk_color_above_ring_encoder.transform(X[["stalk-color-above-ring"]]) 
    # print("stalk_color_above_ring_encoder ", len(stalk_color_above_ring_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    stalk_below_above_ring_encoder = OneHotEncoder()

    # 学习规则
    stalk_below_above_ring_encoder.fit(X[["stalk-color-below-ring"]]) # 9个特征 

    # 转换
    stalk_below_above_ring_transformed = stalk_below_above_ring_encoder.transform(X[["stalk-color-below-ring"]]) 
    # print("stalk_below_above_ring_encoder ", len(stalk_below_above_ring_encoder.categories_[0]))


    # veil_cross = X["veil-type"].astype(str) + "_" + X["veil-color"]
    # # 创建一个包含分类变量的 Series
    # veil_encoder = OneHotEncoder()
    # # # 学习规则
    # veil_encoder.fit(veil_cross.to_frame()) # 4个输入特征
    # # 转换
    # veil_ransformed = veil_encoder.transform(veil_cross.to_frame()) 

    # print("stalk_encoder ", len(veil_encoder.categories_[0]))

    # # 创建一个包含分类变量的 Series
    # veil_type_encoder = OneHotEncoder()

    # # 学习规则
    # veil_type_encoder.fit(X[["veil-type"]]) # 1个特征 可以直接删了

    # # 转换
    # veil_type_transformed = veil_type_encoder.transform(X[["veil-type"]]) 
    # print("veil_type_encoder ", len(veil_type_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    veil_color_encoder = OneHotEncoder()

    # 学习规则
    veil_color_encoder.fit(X[["veil-color"]]) # 4个特征

    # 转换
    veil_color_transformed = veil_color_encoder.transform(X[["veil-color"]]) 
    # print("veil_color_encoder ", len(veil_color_encoder.categories_[0]))

    ring_cross = X["ring-number"].astype(str) + "_" + X["ring-type"]
    # 创建一个包含分类变量的 Series
    ring_encoder = OneHotEncoder()
    # # 学习规则
    ring_encoder.fit(ring_cross.to_frame()) # 7个输入特征
    # 转换
    ring_ransformed = ring_encoder.transform(ring_cross.to_frame()) 
    # print("ring_encoder ", len(ring_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    ring_number_encoder = OneHotEncoder()

    # 学习规则
    ring_number_encoder.fit(X[["ring-number"]]) # 3个特征

    # 转换
    ring_number_transformed = ring_number_encoder.transform(X[["ring-number"]]) 
    # print("ring_number_encoder ", len(ring_number_encoder.categories_[0]))

    # 创建一个包含分类变量的 Series
    ring_type_encoder = OneHotEncoder()

    # 学习规则
    ring_type_encoder.fit(X[["ring-type"]]) # 5个特征

    # 转换
    ring_type_transformed = ring_type_encoder.transform(X[["ring-type"]]) 
    # print("ring_type_encoder ", len(ring_type_encoder.categories_[0]))

    i = (90+6+4+10+34+2+2+2+12+37+2+5+4+4+9+9+4+7+3+5)
    print("i", i)